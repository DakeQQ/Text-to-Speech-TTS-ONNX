import os
import re
import time
import platform
import traceback
import warnings
from typing import List, Union, overload
from functools import lru_cache

import numpy as np
import onnxruntime
import soundfile as sf
from pydub import AudioSegment
from sentencepiece import SentencePieceProcessor

onnx_model_A         = r"/home/DakeQQ/Downloads/IndexTTS_Optimized/IndexTTS_A.onnx"     # The exported onnx model path.
onnx_model_B         = r"/home/DakeQQ/Downloads/IndexTTS_Optimized/IndexTTS_B.onnx"     # The exported onnx model path.
onnx_model_C         = r"/home/DakeQQ/Downloads/IndexTTS_Optimized/IndexTTS_C.onnx"     # The exported onnx model path.
onnx_model_D         = r"/home/DakeQQ/Downloads/IndexTTS_Optimized/IndexTTS_D.onnx"     # The exported onnx model path.
onnx_model_E         = r"/home/DakeQQ/Downloads/IndexTTS_Optimized/IndexTTS_E.onnx"     # The exported onnx model path.
onnx_model_F         = r"/home/DakeQQ/Downloads/IndexTTS_Optimized/IndexTTS_F.onnx"     # The exported onnx model path.
tokenizer_path       = r"/home/DakeQQ/Downloads/IndexTTS-1.5/bpe.model"
generated_audio_path = r"generated.wav"                                            # The generated audio path.
reference_audio      = r"./example/zh.wav"                                         # The reference audio path.
gen_text             = "大家好，我现在正在大可奇奇体验 ai 科技。"                        # The target speech.


# ONNX Runtime Settings
ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
# Model Parameters
SAMPLE_RATE = 24000                     # IndexTTS model setting
STOP_TOKEN = [8193]                     # IndexTTS model setting
MAX_GENERATE_LENGTH = 800               # IndexTTS model setting
REPEAT_PENALITY = 0.9                   # Range from 0.0 to 1.0; "1.0" means no penality.
PENALITY_RANGE = 10                     # Penalizes the most recent output. "10" means the last 10 mel tokens.

# Others
DEVICE_ID = 0
MAX_THREADS = 4


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


# ONNX Runtime settings
if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS,
            'num_streams': 1,
            'enable_opencl_throttling': True,
            'enable_qdq_optimizer': False,                # Enable it carefully
            'disable_dynamic_shapes': False
        }
    ]
    device_type = 'cpu'
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 24 * 1024 * 1024 * 1024,     # 24 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',   # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search': 'EXHAUSTIVE',       # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
            'sdpa_kernel': '2',                           # ["0", "1", "2"]
            'use_tf32': '1',
            'fuse_conv_bias': '1',
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'tunable_op_enable': '1',
            'tunable_op_tuning_enable': '1',
            'tunable_op_max_tuning_duration_ms': 10000,
            'do_copy_in_default_stream': '0',
            'enable_cuda_graph': '0',                     # Set to '0' to avoid potential errors when enabled.
            'prefer_nhwc': '0',
            'enable_skip_layer_norm_strict_mode': '0',
            'use_ep_level_unified_stream': '0',
        }
    ]
    device_type = 'cuda'
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
            'device_filter': 'npu'                         # [any, npu, gpu]
        }
    ]
    device_type = 'dml'
else:
    # Please config by yourself for others providers.
    device_type = 'cpu'
    provider_options = None


# Optimized version
def tokenize_by_CJK_char(line: str, do_upper_case=True) -> str:
    """
    Tokenize a line of text with CJK char.

    Note: All return charaters will be upper case.

    Example:
      input = "你好世界是 hello world 的中文"
      output = "你 好 世 界 是 HELLO WORLD 的 中 文"

    Args:
      line:
        The input text.

    Return:
      A new string tokenize by CJK char.
    """
    # Pre-compiled regex pattern for better performance
    if not hasattr(tokenize_by_CJK_char, '_pattern'):
        tokenize_by_CJK_char._pattern = re.compile(
            r"([\u1100-\u11ff\u2e80-text = text.replace("嗯", "恩").replace("呣", "母")\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
        )

    line = line.strip()
    if not line:
        return ""

    chars = tokenize_by_CJK_char._pattern.split(line)

    if do_upper_case:
        return " ".join(w.strip().upper() for w in chars if w.strip())
    else:
        return " ".join(w.strip() for w in chars if w.strip())


def de_tokenized_by_CJK_char(line: str, do_lower_case=False) -> str:
    """
    Example:
      input = "你 好 世 界 是 HELLO WORLD 的 中 文"
      output = "你好世界是 hello world 的中文"

    do_lower_case:
      input = "SEE YOU!"
      output = "see you!"
    """
    # Pre-compiled regex patterns
    if not hasattr(de_tokenized_by_CJK_char, '_patterns'):
        de_tokenized_by_CJK_char._patterns = {
            'english_word': re.compile(r"([A-Z]+(?:[\s-][A-Z-]+)*)", re.IGNORECASE),
            'sent_placeholder': re.compile(r"^.*?(<sent_(\d+)>)")
        }

    english_word_pattern = de_tokenized_by_CJK_char._patterns['english_word']
    sent_placeholder_pattern = de_tokenized_by_CJK_char._patterns['sent_placeholder']

    # replace english words in the line with placeholders
    english_sents = english_word_pattern.findall(line)
    if not english_sents:
        return "".join(line.split())

    # Use a more efficient replacement strategy
    temp_line = line
    for i, sent in enumerate(english_sents):
        temp_line = temp_line.replace(sent, f"<sent_{i}>", 1)  # Only replace first occurrence

    words = temp_line.split()

    # restore english sentences
    for i in range(len(words)):
        m = sent_placeholder_pattern.match(words[i])
        if m:
            # restore the english word
            placeholder_index = int(m.group(2))
            words[i] = words[i].replace(m.group(1), english_sents[placeholder_index])
            if do_lower_case:
                words[i] = words[i].lower()

    return "".join(words)


class TextNormalizer:
    def __init__(self):
        self.zh_normalizer = None
        self.en_normalizer = None

        # Pre-compile character replacement maps as tuples for better memory efficiency
        char_rep_items = [
            ("：", ","), ("；", ","), (";", ","), ("，", ","), ("。", "."),
            ("！", "!"), ("？", "?"), ("\n", " "), ("·", "-"), ("、", ","),
            ("...", "…"), (",,,", "…"), ("，，，", "…"), ("……", "…"),
            (""", "'"), (""", "'"), ('"', "'"), ("'", "'"), ("'", "'"),
            ("（", "'"), ("）", "'"), ("(", "'"), (")", "'"), ("《", "'"),
            ("》", "'"), ("【", "'"), ("】", "'"), ("[", "'"), ("]", "'"),
            ("—", "-"), ("～", "-"), ("~", "-"), ("「", "'"), ("」", "'"),
            (":", ","),
        ]

        self.char_rep_map = dict(char_rep_items)
        self.zh_char_rep_map = {"$": ".", **self.char_rep_map}

        # Pre-compile regex patterns for better performance
        self._compile_patterns()text = text.replace("嗯", "恩").replace("呣", "母")

    def _compile_patterns(self):
        """Pre-compile all regex patterns used in the class"""
        self.email_pattern = re.compile(r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$")
        self.chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
        self.alpha_pattern = re.compile(r"[a-zA-Z]")
        self.pinyin_tone_pattern = re.compile(r"([bmnpqdfghjklzcsxwy]?h?[aeiouüv]{1,2}[ng]*|ng)([1-5])", re.IGNORECASE)
        self.name_pattern = re.compile(r"[\u4e00-\u9fff]+([-·—][\u4e00-\u9fff]+){1,2}")
        self.jqx_pinyin_pattern = re.compile(r"([jqx])[uü](n|e|an)*(\d)", re.IGNORECASE)

        # Pre-compile replacement patterns
        self._char_replacement_pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
        self._zh_char_replacement_pattern = re.compile("|".join(re.escape(p) for p in self.zh_char_rep_map.keys()))

    @lru_cache(maxsize=1024)
    def match_email(self, email):
        """Cache email matching results"""
        return self.email_pattern.match(email) is not None

    PINYIN_TONE_PATTERN = r"([bmnpqdfghjklzcsxwy]?h?[aeiouüv]{1,2}[ng]*|ng)([1-5])"
    NAME_PATTERN = r"[\u4e00-\u9fff]+([-·—][\u4e00-\u9fff]+){1,2}"

    @lru_cache(maxsize=1024)
    def use_chinese(self, s):
        """Cache language detection results"""
        has_chinese = bool(self.chinese_pattern.search(s))
        has_alpha = bool(self.alpha_pattern.search(s))
        is_email = self.match_email(s)

        if has_chinese or not has_alpha or is_email:
            return True

        has_pinyin = bool(self.pinyin_tone_pattern.search(s))
        return has_pinyin

    def load(self):
        if platform.system() == "Darwin":
            from wetext import Normalizer
            self.zh_normalizer = Normalizer(remove_erhua=False, lang="zh", operator="tn")
            self.en_normalizer = Normalizer(lang="en", operator="tn")
        else:
            from tn.chinese.normalizer import Normalizer as NormalizerZh
            from tn.english.normalizer import Normalizer as NormalizerEn
            self.zh_normalizer = NormalizerZh(remove_interjections=False, remove_erhua=False, overwrite_cache=False)
            self.en_normalizer = NormalizerEn(overwrite_cache=False)

    def normalize(self, text: str) -> str:
        text = text.replace("嗯", "恩").replace("呣", "母")
        if not self.zh_normalizer or not self.en_normalizer:
            print("Error, text normalizer is not initialized !!!")
            return ""

        text = text.rstrip()

        if self.use_chinese(text):
            replaced_text, pinyin_list = self.save_pinyin_tones(text)
            replaced_text, original_name_list = self.save_names(replaced_text)

            try:
                result = self.zh_normalizer.normalize(replaced_text)
            except Exception:
                result = ""
                print(traceback.format_exc())

            # Restore in reverse order
            result = self.restore_names(result, original_name_list)
            result = self.restore_pinyin_tones(result, pinyin_list)
            result = self._zh_char_replacement_pattern.sub(lambda x: self.zh_char_rep_map[x.group()], result)
        else:
            try:
                result = self.en_normalizer.normalize(text)
            except Exception:
                result = text
                print(traceback.format_exc())
            result = self._char_replacement_pattern.sub(lambda x: self.char_rep_map[x.group()], result)

        return result

    def correct_pinyin(self, pinyin: str):
        """Optimized pinyin correction with pre-compiled regex"""
        if pinyin[0].lower() not in "jqx":
            return pinyin

        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = self.jqx_pinyin_pattern.sub(repl, pinyin)
        return pinyin.uppeOthers
DEVICE_ID = 0
MAX_THREADS = 4


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


# ONNX Runtime settings
if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS,
            'num_streams': 1,
            'enable_opencl_throttling': True,
            'enable_qdq_optimizer': False,                # Enable it carefully
            'disable_dynamic_shapes': False
        }
    ]
    device_type = 'cpu'
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 24 * 1024 * 1024 * 1024,     # 24 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',   # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search': 'EXHAUSTIVE',       # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
            'sdpa_kernel': '2',                           # ["0", "1", "2"]
            'use_tf32': '1',
            'fuse_conv_bias': '1',
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'tunable_op_enable': '1',
            'tunable_op_tuning_enable': '1',
            'tunable_op_max_tuning_duration_ms': 10000,
            'do_copy_in_default_stream': '0',
            'enable_cuda_graph': '0',                     # Set to '0' to avoid potential errors when enabled.
            'prefer_nhwc': '0',
            'enable_skip_layer_norm_strict_mode': '0',
            'use_ep_level_unified_stream': '0',
        }
    ]
    device_type = 'cuda'
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.
r()text = text.replace("嗯", "恩").replace("呣", "母")

    def save_names(self, original_text):
        """Optimized name saving with early returns"""
        original_name_list = self.name_pattern.findall(original_text)
        if not original_name_list:
            return original_text, None

        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in ("".join(n) for n in original_name_list):
            if name not in seen:
                seen.add(name)
                unique_names.append(name)

        if not unique_names:
            return original_text, None

        transformed_text = original_text
        for i, name in enumerate(unique_names):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(name, f"<n_{number}>")

        return transformed_text, unique_names

    def restore_names(self, normalized_text, original_name_list):
        """Optimized name restoration with early return"""
        if not original_name_list:
            return normalized_text

        transformed_text = normalized_text
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(f"<n_{number}>", name)
        return transformed_text

    def save_pinyin_tones(self, original_text):
        """Optimized pinyin saving with early returns"""
        original_pinyin_list = self.pinyin_tone_pattern.findall(original_text)
        if not original_pinyin_list:
            return original_text, None

        # Remove duplicates while preserving order
        seen = set()
        unique_pinyins = []
        for pinyin in ("".join(p) for p in original_pinyin_list):
            if pinyin not in seen:
                seen.add(pinyin)
                unique_pinyins.append(pinyin)

        if not unique_pinyins:
            return original_text, None

        transformed_text = original_text
        for i, pinyin in enumerate(unique_pinyins):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(pinyin, f"<pinyin_{number}>")

        return transformed_text, unique_pinyins

    def restore_pinyin_tones(self, normalized_text, original_pinyin_list):
        """Optimized pinyin restoration with early return"""
        if not original_pinyin_list:
            return normalized_text

        transformed_text = normalized_text
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            corrected_pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f"<pinyin_{number}>", corrected_pinyin)
        return transformed_text


class TextTokenizer:
    def __init__(self, vocab_file: str, normalizer: TextNormalizer = None):
        self.vocab_file = vocab_file
        self.normalizer = normalizer

        if self.vocab_file is None:
            raise ValueError("vocab_file is None")
        if not os.path.exists(self.vocab_file):
            raise ValueError(f"vocab_file {self.vocab_file} does not exist")

        if self.normalizer:
            self.normalizer.load()

        # Load vocabulary
        self.sp_model = SentencePieceProcessor(model_file=self.vocab_file)

        self.pre_tokenizers = [tokenize_by_CJK_char]

        # Cache frequently used properties
        self._vocab_size = self.sp_model.GetPieceSize()
        self._unk_token_id = self.sp_model.unk_id()

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def unk_token(self):
        return "<unk>"

    @property
    def pad_token(self):
        return None

    @property
    def bos_token(self):
        return "<s>"

    @property
    def eos_token(self):
        return "</s>"

    @property
    def pad_token_id(self):
        return -1

    @property
    def bos_token_id(self):
        return 0

    @property
    def eos_token_id(self):
        return 1

    @property
    def unk_token_id(self):
        return self._unk_token_id

    @property
    def special_tokens_map(self):
        return {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }

    @lru_cache(maxsize=102400)
    def get_vocab(self):
        """Cache vocabulary dictionary"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    @overload
    def convert_ids_to_tokens(self, ids: int) -> str:
        ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        ...

    def convert_ids_to_tokens(self, ids: Union[List[int], int]):
        return self.sp_model.IdToPiece(ids)

    def convert_tokens_to_ids(self, tokens: Union[List[str], str]) -> List[int]:
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self.sp_model.PieceToId(token) for token in tokens]

    def tokenize(self, text: str) -> List[str]:
        return self.encode(text, out_type=str)

    def encode(self, text: str, **kwargs):
        if not text:
            return []

        text_stripped = text.strip()
        if len(text_stripped) == 1:
            return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)

        # Preprocessing
        processed_text = text
        if self.normalizer:
            processed_text = self.normalizer.normalize(processed_text)

        # Apply pre-tokenizers
        for pre_tokenizer in self.pre_tokenizers:
            processed_text = pre_tokenizer(processed_text)

        return self.sp_model.Encode(processed_text, out_type=kwargs.pop("out_type", int), **kwargs)

    def batch_encode(self, texts: List[str], **kwargs):
        """Optimized batch processing"""
        processed_texts = texts

        # Batch normalization
        if self.normalizer:
            processed_texts = [self.normalizer.normalize(text) for text in processed_texts]

        # Batch pre-tokenization
        for pre_tokenizer in self.pre_tokenizers:
            processed_texts = [pre_tokenizer(text) for text in processed_texts]

        return self.sp_model.Encode(processed_texts, out_type=kwargs.pop("out_type", int), **kwargs)

    def decode(self, ids: Union[List[int], int], do_lower_case=False, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        decoded = self.sp_model.Decode(ids, out_type=kwargs.pop("out_type", str), **kwargs)
        return de_tokenized_by_CJK_char(decoded, do_lower_case=do_lower_case)

    @staticmethod
    def split_sentences_by_token(
            tokenized_str: List[str], split_tokens: List[str], max_tokens_per_sentence: int
    ) -> List[List[str]]:
        """Optimized sentence splitting with early returns and reduced operations"""
        if not tokenized_str:
            return []

        sentences: List[List[str]] = []
        current_sentence = []
        split_tokens_set = set(split_tokens)  # Convert to set for O(1) lookup

        i = 0
        while i < len(tokenized_str):
            token = tokenized_str[i]
            current_sentence.append(token)

            if token in split_tokens_set:
                # Skip empty or minimal sentences
                if len(current_sentence) <= 1 or (len(current_sentence) == 2 and current_sentence[0] == '▁'):
                    current_sentence = []
                    i += 1
                    continue

                # Handle quote continuation
                if i < len(tokenized_str) - 1 and tokenized_str[i + 1] in ["'", "▁'"]:
                    current_sentence.append(tokenized_str[i + 1])
                    i += 1

                if len(current_sentence) <= max_tokens_per_sentence:
                    sentences.append(current_sentence)
                else:
                    # Handle oversized sentences
                    sub_sentences = TextTokenizer._handle_oversized_sentence(
                        current_sentence, max_tokens_per_sentence
                    )
                    sentences.extend(sub_sentences)

                current_sentence = []
            i += 1

        if current_sentence:
            sentences.append(current_sentence)

        # Merge adjacent short sentences
        return TextTokenizer._merge_short_sentences(sentences, max_tokens_per_sentence)

    @staticmethod
    def _handle_oversized_sentence(sentence: List[str], max_tokens: int) -> List[List[str]]:
        """Handle sentences that exceed the maximum token limit"""
        if "," in sentence or "▁," in sentence:
            return TextTokenizer.split_sentences_by_token(sentence, [",", "▁,"], max_tokens)
        elif "-" in sentence:
            return TextTokenizer.split_sentences_by_token(sentence, ["-"], max_tokens)
        else:
            warnings.warn(
                f"The tokens length of sentence exceeds limit: {max_tokens}, "
                f"Tokens in sentence: {sentence}. Maybe unexpected behavior",
                RuntimeWarning,
            )
            return [sentence[:max_tokens], sentence[max_tokens:]]

    @staticmethod
    def _merge_short_sentences(sentences: List[List[str]], max_tokens: int) -> List[List[str]]:
        """Merge adjacent sentences if they fit within the token limit"""
        if not sentences:
            return []

        merged_sentences = []
        for sentence in sentences:
            if not sentence:
                continue

            if not merged_sentences:
                merged_sentences.append(sentence)
            elif len(merged_sentences[-1]) + len(sentence) <= max_tokens:
                merged_sentences[-1].extend(sentence)
            else:
                merged_sentences.append(sentence)

        return merged_sentences

    # Class variable for better performance
    punctuation_marks_tokens = [
        ".", "!", "?", "▁.", "▁?", "▁...",
    ]

    def split_sentences(self, tokenized: List[str], max_tokens_per_sentence=120) -> List[List[str]]:
        return TextTokenizer.split_sentences_by_token(
            tokenized, self.punctuation_marks_tokens, max_tokens_per_sentence=max_tokens_per_sentence
        )


normalizer = TextNormalizer()
normalizer.load()
tokenizer = TextTokenizer(tokenizer_path, normalizer)
del normalizer

session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4         # Fatal level = 4, it an adjustable value.
session_opts.log_verbosity_level = 4        # Fatal level = 4, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
amount_of_outputs_A = len(out_name_A)
out_name_A = [out_name_A[i].name for i in range(amount_of_outputs_A)]
last_output_indices_A = amount_of_outputs_A - 1


ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
in_name_B0 = in_name_B[0].name
out_name_B0 = out_name_B[0].name


ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
in_name_C0 = in_name_C[0].name
in_name_C1 = in_name_C[1].name
out_name_C0 = out_name_C[0].name
out_name_C1 = out_name_C[1].name


ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_D = ort_session_D.get_inputs()
out_name_D = ort_session_D.get_outputs()
in_name_D0 = in_name_D[0].name
in_name_D1 = in_name_D[1].name
out_name_D0 = out_name_D[0].name
out_name_D1 = out_name_D[1].name


ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
print(f"\nUsable Providers: {ort_session_E.get_providers()[0]}")
model_E_dtype = ort_session_E._inputs_meta[0].type
if 'float16' in model_E_dtype:
    model_E_dtype = np.float16
else:
    model_E_dtype = np.float32
in_names_E = ort_session_E.get_inputs()
out_name_E = ort_session_E.get_outputs()
amount_of_inputs_E = len(in_names_E)
amount_of_outputs_E = len(out_name_E)
in_names_E = [in_names_E[i].name for i in range(amount_of_inputs_E)]
out_name_E = [out_name_E[i].name for i in range(amount_of_outputs_E)]
num_layers = (amount_of_outputs_E - 3) // 2
num_layers_2 = num_layers + num_layers
num_layers_2_plus_1 = num_layers_2 + 1
num_layers_2_plus_2 = num_layers_2 + 2
num_layers_2_plus_3 = num_layers_2 + 3
last_input_indices_E = amount_of_inputs_E - 1
last_output_indices_E = amount_of_outputs_E - 1
second_last_output_indices_E = amount_of_outputs_E - 2


ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_F = ort_session_F.get_inputs()
out_name_F = ort_session_F.get_outputs()
in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
out_name_F0 = out_name_F[0].name


# Run IndexTTS by ONNX Runtime
audio = np.array(AudioSegment.from_file(reference_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
audio = normalize_to_int16(audio)
audio = audio.reshape(1, 1, -1)
audio = onnxruntime.OrtValue.ortvalue_from_numpy(audio, device_type, DEVICE_ID)
init_gpt_ids = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[8192]], dtype=np.int32), device_type, DEVICE_ID)
init_gen_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
init_ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
init_history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
init_attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)
init_attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)
if device_type != 'dml':
    init_past_keys_E = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_E._inputs_meta[0].shape[0], ort_session_E._inputs_meta[0].shape[1], 0), dtype=model_E_dtype), device_type, DEVICE_ID)
    init_past_values_E = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_E._inputs_meta[num_layers].shape[0], 0, ort_session_E._inputs_meta[num_layers].shape[2]), dtype=model_E_dtype), device_type, DEVICE_ID)
else:
    init_past_keys_E = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_E._inputs_meta[0].shape[0], ort_session_E._inputs_meta[0].shape[1], 0), dtype=model_E_dtype), 'cpu', DEVICE_ID)
    init_past_values_E = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_E._inputs_meta[num_layers].shape[0], 0, ort_session_E._inputs_meta[num_layers].shape[2]), dtype=model_E_dtype), 'cpu', DEVICE_ID)
repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((1, ort_session_E._inputs_meta[num_layers_2_plus_1].shape[1]), dtype=np.float32), device_type, DEVICE_ID)
split_pad = np.zeros((1, 1, int(SAMPLE_RATE * 0.2)), dtype=np.int16)  # Default to 200ms split padding.


input_feed_E = {
    in_names_E[last_input_indices_E]: init_attention_mask_1,
    in_names_E[num_layers_2]: init_history_len,
    in_names_E[num_layers_2_plus_1]: repeat_penality
}
for i in range(num_layers):
    input_feed_E[in_names_E[i]] = init_past_keys_E
for i in range(num_layers, num_layers_2):
    input_feed_E[in_names_E[i]] = init_past_values_E


text_tokens_list = tokenizer.tokenize(gen_text)
sentences = tokenizer.split_sentences(text_tokens_list)
total_sentences = len(sentences)
save_generated_wav = []

# Start to Run IndexTTS
start_time = time.time()

all_outputs_A = ort_session_A.run_with_ort_values(
    out_name_A,
    {
        in_name_A0: audio
    })

input_feed_F = {}
for i in range(last_output_indices_A):
    input_feed_F[in_name_F[i]] = all_outputs_A[i]

for i in range(total_sentences):
    sent = sentences[i]
    split_text = "".join(sent).replace("▁", " ")
    print(f"\nGenerate the Voice for '{split_text}'")

    text_tokens = tokenizer.convert_tokens_to_ids(sent)
    text_ids = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([text_tokens], dtype=np.int32), device_type, DEVICE_ID)

    text_hidden_state = ort_session_B.run_with_ort_values(
        [out_name_B0],
        {
            in_name_B0: text_ids,
        })[0]

    mel_emb = ort_session_D.run_with_ort_values(
        [out_name_D0],
        {
            in_name_D0: all_outputs_A[last_output_indices_A],
            in_name_D1: text_hidden_state
        })[0]

    gpt_hidden_state, gen_len = ort_session_C.run_with_ort_values(
        [out_name_C0, out_name_C1],
        {
            in_name_C0: init_gpt_ids,
            in_name_C1: init_gen_len
        })

    gpt_hidden_state, concat_len = ort_session_D.run_with_ort_values(
        [out_name_D0, out_name_D1],
        {
            in_name_D0: mel_emb,
            in_name_D1: gpt_hidden_state
        })

    generate_limit = MAX_GENERATE_LENGTH - onnxruntime.OrtValue.numpy(concat_len)
    input_feed_E[in_names_E[num_layers_2_plus_2]] = concat_len

    save_last_hidden_state = []
    save_max_logits_ids = []
    reset_penality = 0
    num_decode = 0
  
    decode_time = time.time()
    while num_decode < generate_limit:
        input_feed_E[in_names_E[num_layers_2_plus_3]] = gpt_hidden_state
        all_outputs_E = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
        max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_E[last_output_indices_E])
        save_max_logits_ids.append(max_logit_ids)
        save_last_hidden_state.append(all_outputs_E[second_last_output_indices_E])
        num_decode += 1
        if max_logit_ids in STOP_TOKEN:
            break
        if num_decode < 2:
            input_feed_E[in_names_E[last_input_indices_E]] = init_attention_mask_0
            input_feed_E[in_names_E[num_layers_2_plus_2]] = init_ids_len_1
        for i in range(second_last_output_indices_E):
            input_feed_E[in_names_E[i]] = all_outputs_E[i]
        repeat_penality = onnxruntime.OrtValue.numpy(repeat_penality)
        repeat_penality[:, max_logit_ids] = REPEAT_PENALITY
        if (num_decode > PENALITY_RANGE) and (save_max_logits_ids[reset_penality] != max_logit_ids):
            repeat_penality[:, save_max_logits_ids[reset_penality]] = 1.0
            reset_penality += 1
        repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(repeat_penality, device_type, DEVICE_ID)
        gpt_hidden_state, gen_len = ort_session_C.run_with_ort_values(
            [out_name_C0, out_name_C1],
            {
                in_name_C0: all_outputs_E[last_output_indices_E],
                in_name_C1: gen_len
            })
    print(f"\n\nDecode Speed: {num_decode / (time.time() - decode_time):.3f} tokens/s")

    for i in range(num_decode):
         save_last_hidden_state[i] = onnxruntime.OrtValue.numpy(save_last_hidden_state[i])
    input_feed_F[in_name_F[last_output_indices_A]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.concatenate(save_last_hidden_state, axis=0), device_type, DEVICE_ID)
    
    generated_wav = ort_session_F.run_with_ort_values(
        [out_name_F0],
        input_feed_F
    )[0]
    generated_wav = np.concatenate([onnxruntime.OrtValue.numpy(generated_wav), split_pad], axis=-1)

    # Init
    input_feed_E[in_names_E[last_input_indices_E]] = init_attention_mask_1
    input_feed_E[in_names_E[num_layers_2]] = init_history_len
    for i in range(num_layers):
        input_feed_E[in_names_E[i]] = init_past_keys_E
    for i in range(num_layers, num_layers_2):
        input_feed_E[in_names_E[i]] = init_past_values_E


# Save to audio
sf.write(generated_audio_path, generated_wav.reshape(-1), SAMPLE_RATE, format='WAVEX')
print(f"\nAudio generation is complete.\n\nONNXRuntime Time Cost in Seconds:\n{time.time() - start_time:.3f}")
