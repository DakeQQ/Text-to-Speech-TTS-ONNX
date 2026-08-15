"""Run the optimized Inflect v2 ONNX package without importing PyTorch."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import soundfile as sf
from num2words import num2words

from Shared_Weights import attach_shared_initializers


SCRIPT_DIR = Path(__file__).resolve().parent
METADATA_MODEL_NAME = "Inflect_Metadata.onnx"


# ============================== Configuration ==============================
# Edit these values directly; the CLI is reserved for selecting the ONNX folder.
TARGET_TEXT = "Hello everyone, I'm currently experiencing DakeQQ's AI technology."
GENERATED_AUDIO_PATH = SCRIPT_DIR / "generated.wav"

SPEED = 1.0                        # Speaking speed from 0.5 to 2.0.
VARIATION = 0.6                    # Acoustic variation from 0.0 to 1.0.
RANDOM_SEED = 9527                 # Seed used to initialize the graph RNG.
MAX_CHUNK_LENGTH = 280             # Maximum normalized characters per chunk.

ORT_LOG = False                    # Enable ONNX Runtime logging.
ORT_FP16 = False                   # Preserve FP16 runtime paths where supported.
ORT_ACCELERATE_PROVIDERS = []      # Optional providers, e.g. ['CUDAExecutionProvider'].
MAX_THREADS = 0                    # CPU parallel threads; 0 lets ORT choose.
DEVICE_ID = 0                      # Accelerator device index.
SHOW_PROGRESS = True               # Print pipeline stages and synthesis progress.
# ===========================================================================


def print_progress(message):
	if SHOW_PROGRESS:
		print(f"[Inflect] {message}", flush=True)


def parse_args():
	parser = argparse.ArgumentParser(description="Run compact Inflect ONNX inference.")
	parser.add_argument(
		"--onnx-folder",
		"--model-folder",
		dest="onnx_folder",
		type=Path,
		default=SCRIPT_DIR / "Inflect_Optimized",
		help="Folder containing the compact Inflect ONNX package.",
	)
	return parser.parse_args()


ARGS = parse_args()
ONNX_FOLDER = ARGS.onnx_folder.expanduser().resolve()
GENERATED_AUDIO_PATH = GENERATED_AUDIO_PATH.expanduser().resolve()
MONTHS = (
	"January", "February", "March", "April", "May", "June", "July",
	"August", "September", "October", "November", "December",
)
WORD_OVERRIDES = {
	"24/7": "twenty four seven",
	"Airbnb": "air bee en bee",
	"ASAP": "as soon as possible",
	"BRB": "be right back",
	"BTW": "by the way",
	"DIY": "do it yourself",
	"DMs": "direct messages",
	"DM": "direct message",
	"eBay": "ee bay",
	"ETA": "estimated time of arrival",
	"Facebook": "face book",
	"FAQ": "frequently asked questions",
	"FYI": "for your information",
	"IDK": "I don't know",
	"IMO": "in my opinion",
	"IMHO": "in my humble opinion",
	"Instagram": "in sta gram",
	"iPad": "eye pad",
	"iPhone": "eye phone",
	"IRL": "in real life",
	"LEGO": "leg oh",
	"LOL": "laughing out loud",
	"Netflix": "net flicks",
	"Nike": "nye key",
	"OMG": "oh my goodness",
	"PayPal": "pay pal",
	"RSVP": "please respond",
	"selfie": "selfie",
	"SMH": "shaking my head",
	"Snapchat": "snap chat",
	"Spotify": "spot if eye",
	"TBH": "to be honest",
	"TikTok": "tick tock",
	"TMI": "too much information",
	"Uber": "oo ber",
	"URL": "you are ell",
	"Wi-Fi": "why fie",
	"WiFi": "why fie",
	"WhatsApp": "what's app",
	"YOLO": "you only live once",
	"YouTube": "you tube",
}
LETTER_NAMES = {
	"A": "ay", "B": "bee", "C": "see", "D": "dee", "E": "ee", "F": "eff",
	"G": "gee", "H": "aitch", "I": "eye", "J": "jay", "K": "kay", "L": "ell",
	"M": "em", "N": "en", "O": "oh", "P": "pee", "Q": "cue", "R": "ar",
	"S": "ess", "T": "tee", "U": "you", "V": "vee", "W": "double you",
	"X": "ex", "Y": "why", "Z": "zee",
}
LETTER_PLURALS = {
	"A": "ays", "B": "bees", "C": "sees", "D": "dees", "E": "ees",
	"F": "effs", "G": "gees", "H": "aitches", "I": "eyes", "J": "jays",
	"K": "kays", "L": "ells", "M": "ems", "N": "ens", "O": "ohs",
	"P": "pees", "Q": "cues", "R": "ars", "S": "esses", "T": "tees",
	"U": "yous", "V": "vees", "W": "double yous", "X": "exes",
	"Y": "whys", "Z": "zees",
}
ABBREVIATIONS = {
	"a.k.a.": "also known as", "acct.": "account", "addr.": "address",
	"approx.": "approximately",
	"appt.": "appointment", "Apr.": "April", "Aug.": "August",
	"Ave.": "avenue", "avg.": "average", "bldg.": "building",
	"Blvd.": "boulevard", "Cir.": "circle", "Co.": "company",
	"Ct.": "court", "Dec.": "December", "Dept.": "department",
	"Dr.": "doctor", "Drs.": "doctors", "e.g.": "for example",
	"etc.": "et cetera", "Feb.": "February", "Fri.": "Friday",
	"govt.": "government", "Hon.": "honorable", "Hosp.": "hospital",
	"hr.": "hour", "hrs.": "hours", "Hwy.": "highway",
	"Inc.": "incorporated",
	"Jan.": "January", "Jr.": "junior", "Jul.": "July",
	"Jun.": "June", "Ln.": "lane", "Ltd.": "limited",
	"Mar.": "March", "max.": "maximum", "Mon.": "Monday",
	"Mr.": "mister", "Mrs.": "missus", "Ms.": "miss",
	"Mt.": "mount", "Mtn.": "mountain", "Mx.": "mixter",
	"Nov.": "November",
	"Oct.": "October", "Ofc.": "officer", "org.": "organization",
	"P.S.": "postscript", "Pkwy.": "parkway", "Pl.": "place",
	"qty.": "quantity", "R.S.V.P.": "please respond", "Rd.": "road",
	"Rm.": "room", "Rte.": "route", "Sat.": "Saturday",
	"Sep.": "September", "Sept.": "September", "Sr.": "senior",
	"Ste.": "suite", "Sun.": "Sunday", "Ter.": "terrace",
	"Thu.": "Thursday", "Thur.": "Thursday", "Thurs.": "Thursday",
	"Tue.": "Tuesday", "Tues.": "Tuesday", "Univ.": "university",
	"vol.": "volume", "vs.": "versus", "Wed.": "Wednesday",
}
PUNCT_TRANSLATION = str.maketrans({
	"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
	"\u2013": "-", "\u2014": ", ", "\u2026": "...", "(": ", ", ")": ", ",
	"[": ", ", "]": ", ", "{": ", ", "}": ", ",
})
_ESPEAK_CONFIGURED = False
_ESPEAK_BACKEND = None
_ESPEAK_LOCK = threading.Lock()
_ORT_SEED_LOCK = threading.Lock()


def _words(value: int | float, *, ordinal: bool = False) -> str:
	return num2words(value, to="ordinal" if ordinal else "cardinal").replace("-", " ").replace(",", "")


def _digit_words(text: str) -> str:
	return " ".join(_words(int(character)) for character in text if character.isdigit())


def _identifier_digits(text: str) -> str:
	return " ".join(
		"oh" if character == "0" and index > 0 else _words(int(character))
		for index, character in enumerate(text) if character.isdigit()
	)


def _expand_identifier_token(token: str) -> str:
	match = re.fullmatch(r"([A-Za-z]?)(\d+)([A-Za-z]?)", token)
	if match is None:
		return token
	prefix, digits, suffix = match.groups()
	pieces = []
	if prefix:
		pieces.append(LETTER_NAMES[prefix.upper()])
	pieces.append(_identifier_digits(digits) if len(digits) == 3 or digits.startswith("0") else _words(int(digits)))
	if suffix:
		pieces.append(LETTER_NAMES[suffix.upper()])
	return " ".join(pieces)


def _expand_money(match: re.Match[str]) -> str:
	dollars, _, cents = match.group(1).replace(",", "").partition(".")
	count = int(dollars)
	pieces = [_words(count), "dollar" if count == 1 else "dollars"]
	if cents and (cent_count := int(cents[:2].ljust(2, "0"))):
		pieces.extend(("and", _words(cent_count), "cent" if cent_count == 1 else "cents"))
	return " ".join(pieces)


def _expand_date(match: re.Match[str]) -> str:
	from datetime import date
	month, day, year = (int(value) for value in match.groups())
	try:
		date(year, month, day)
	except ValueError:
		return match.group(0)
	return f"{MONTHS[month - 1]} {_words(day, ordinal=True)} {_words(year)}"


def _expand_time(match: re.Match[str]) -> str:
	hour, minute, suffix = int(match.group(1)), int(match.group(2)), match.group(3) or ""
	pieces = [_words(hour)]
	pieces.extend(("o clock",) if minute == 0 else (("oh", _words(minute)) if minute < 10 else (_words(minute),)))
	if suffix:
		pieces.extend(re.sub(r"[^A-Za-z]", "", suffix).lower())
	return " ".join(pieces)


def _replace_literals(text: str, replacements: dict[str, str], *, ignore_case: bool = False) -> str:
	flags = re.IGNORECASE if ignore_case else 0
	for source in sorted(replacements, key=len, reverse=True):
		pattern = rf"(?<!\w){re.escape(source)}(?!\w)"
		text = re.sub(pattern, replacements[source], text, flags=flags)
	return text


def _expand_plural_acronym(match: re.Match[str]) -> str:
	acronym = match.group(1)
	return " ".join(
		[*(LETTER_NAMES[character] for character in acronym[:-1]), LETTER_PLURALS[acronym[-1]]]
	)


@lru_cache(maxsize=256)
def normalize_text(text: str) -> str:
	text = re.sub(r"\s+", " ", text.translate(PUNCT_TRANSLATION)).strip()
	text = _replace_literals(text, WORD_OVERRIDES)
	text = _replace_literals(text, ABBREVIATIONS, ignore_case=True)
	text = re.sub(r"\b([A-Z])(?:\.([A-Z]))+\.", lambda match: " ".join(re.findall(r"[A-Z]", match.group(0))), text)
	text = re.sub(r"\b(apartment|apt\.?|suite|unit|room|flight|extension|order|invoice|locker|aisle|gate)\s+([A-Za-z]?\d{1,4}[A-Za-z]?)\b", lambda match: f"{match.group(1)} {_expand_identifier_token(match.group(2))}", text, flags=re.IGNORECASE)
	text = re.sub(r"\b(\d{3})(?=\s+(?:North|South|East|West)\b)", lambda match: _identifier_digits(match.group(1)), text, flags=re.IGNORECASE)
	text = re.sub(r"\$(\d[\d,]*(?:\.\d{1,2})?)", _expand_money, text)
	text = re.sub(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(20\d{2}|19\d{2})\b", _expand_date, text)
	text = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?\s*[Mm]\.?)?\b", _expand_time, text)
	text = re.sub(r"\b(\d{1,2})\s*([AaPp]\.?\s*[Mm]\.?)\b", lambda match: f"{_words(int(match.group(1)))} {' '.join(re.sub(r'[^A-Za-z]', '', match.group(2)).lower())}", text)
	text = re.sub(r"\b(\d{3})-(\d{4})\b", lambda match: f"{_digit_words(match.group(1))}, {_digit_words(match.group(2))}", text)
	text = re.sub(r"\b\d+(?:\.\d+){2,}\b", lambda match: " point ".join(_words(int(part)) for part in match.group(0).split(".")), text)
	text = re.sub(r"\b(\d+)\.(\d+)\b", lambda match: f"{_words(int(match.group(1)))} point {_digit_words(match.group(2))}", text)
	text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda match: _words(int(match.group(1)), ordinal=True), text, flags=re.IGNORECASE)
	text = re.sub(r"\b\d[\d,]*\b", lambda match: _digit_words(match.group(0).replace(",", "")) if len(match.group(0).replace(",", "")) >= 5 and not match.group(0).startswith("20") else _words(int(match.group(0).replace(",", ""))), text)
	text = re.sub(r"\b([A-Z]{2,})s\b", _expand_plural_acronym, text)
	text = re.sub(r"\b[A-Z]{2,}\b", lambda match: " ".join(LETTER_NAMES.get(character, character) for character in match.group(0)), text)
	text = re.sub(r",(?:\s*,)+", ",", text)
	text = re.sub(r",\s*([.!?])", r"\1", text)
	text = re.sub(r"\s+([,;:.!?])", r"\1", text)
	return re.sub(r"\s+", " ", re.sub(r"([,;:.!?])(?=\S)", r"\1 ", text)).strip()


def _configure_espeak() -> None:
	global _ESPEAK_CONFIGURED
	if _ESPEAK_CONFIGURED:
		return
	paths = (Path("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1"), Path("/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1"), Path("/usr/lib64/libespeak-ng.so.1"))
	system_library = next((path for path in paths if path.is_file()), None)
	if system_library is not None:
		os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(system_library))
	else:
		import espeakng_loader
		os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path())
		os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
		espeakng_loader.make_library_available()
		espeakng_loader.load_library()
	_ESPEAK_CONFIGURED = True


@lru_cache(maxsize=256)
def _phonemize_normalized(normalized: str) -> str:
	global _ESPEAK_BACKEND
	with _ESPEAK_LOCK:
		_configure_espeak()
		if _ESPEAK_BACKEND is None:
			from phonemizer.backend import EspeakBackend
			_ESPEAK_BACKEND = EspeakBackend(
				language="en-us",
				preserve_punctuation=True,
				with_stress=True,
			)
		phonemes = _ESPEAK_BACKEND.phonemize(
			[normalized],
			strip=True,
			njobs=1,
		)[0]
	return re.sub(r"\s+", " ", phonemes).strip().replace("sˈæskɐtʃˌuːən", "sɐskˈætʃəwən").replace("flʊɹɹˈɛsənt", "flʊˈɹɛsənt")


def frontend_phonemes(text: str) -> str:
	return _phonemize_normalized(normalize_text(text))


def split_text(text: str, limit: int = MAX_CHUNK_LENGTH) -> list[str]:
	normalized = " ".join(text.split())
	sentences = [
		part.strip()
		for part in re.split(r"(?<=[.!?;:])\s+", normalized)
		if part.strip()
	]
	chunks: list[str] = []
	for sentence in sentences or [normalized]:
		while len(sentence) > limit:
			search = sentence[: limit + 1]
			punctuation = max(search.rfind(mark) for mark in (",", ";", ":"))
			split_at = (
				punctuation + 1
				if punctuation >= limit // 2
				else sentence.rfind(" ", 0, limit + 1)
			)
			if split_at < limit // 2:
				split_at = limit
			chunks.append(sentence[:split_at].strip())
			sentence = sentence[split_at:].strip()
		if sentence:
			chunks.append(sentence)
	return chunks


BOUNDARY_PAUSE_SECONDS = {
	"?": 0.28,
	"!": 0.24,
	".": 0.22,
	";": 0.16,
	":": 0.13,
	",": 0.09,
}
DEFAULT_BOUNDARY_PAUSE_SECONDS = 0.08


def boundary_pause_seconds(chunk: str) -> float:
	ending = chunk.rstrip()[-1:] if chunk.strip() else ""
	return BOUNDARY_PAUSE_SECONDS.get(ending, DEFAULT_BOUNDARY_PAUSE_SECONDS)


def read_metadata(path: Path) -> dict[str, str]:
	model = onnx.load(str(path), load_external_data=False)
	return {property_.key: property_.value for property_ in model.metadata_props}


def require_metadata(metadata: dict[str, str], key: str) -> str:
	try:
		return metadata[key]
	except KeyError as exc:
		raise ValueError(f"Missing required Inflect metadata key: {key!r}.") from exc
def io_dtype(argument: object) -> np.dtype:
	match = re.fullmatch(r"tensor\(([^)]+)\)", argument.type)
	element_type = onnx.TensorProto.DataType.Value(match.group(1).upper())
	return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(element_type))


def static_io_dimension(argument: object, axis: int) -> int:
	try:
		dimension = argument.shape[axis]
	except IndexError as exc:
		raise ValueError(
			f"ONNX value {argument.name!r} has no dimension at axis {axis}."
		) from exc
	return dimension


def configure_provider():
	if "OpenVINOExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
		provider_options = [{
			"device_type": "CPU",
			"precision": "ACCURACY",
			"num_of_threads": MAX_THREADS if MAX_THREADS else 8,
			"num_streams": 1,
			"enable_opencl_throttling": False,
			"enable_qdq_optimizer": False,
			"disable_dynamic_shapes": False,
		}]
		device_type = "cpu"
	elif "CUDAExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
		provider_options = [{
			"device_id": DEVICE_ID,
			"gpu_mem_limit": 24 * 1024**3,
			"arena_extend_strategy": "kNextPowerOfTwo",
			"cudnn_conv_algo_search": "EXHAUSTIVE",
			"use_tf32": "1",
			"do_copy_in_default_stream": "0",
			"enable_cuda_graph": "0",
		}]
		device_type = "cuda"
	elif "DmlExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
		provider_options = [{
			"device_id": DEVICE_ID,
			"performance_preference": "high_performance",
			"device_filter": "gpu",
			"disable_metacommands": "false",
			"enable_graph_capture": "false",
		}]
		device_type = "dml"
	else:
		provider_options = None
		device_type = "cpu"
	return provider_options, device_type


def build_session_options():
	session_options = onnxruntime.SessionOptions()
	run_options = onnxruntime.RunOptions()
	for options in (session_options, run_options):
		options.log_severity_level = 0 if ORT_LOG else 4
		options.log_verbosity_level = 4
	session_options.inter_op_num_threads = MAX_THREADS
	session_options.intra_op_num_threads = MAX_THREADS
	session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
	session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
	session_options.enable_cpu_mem_arena = True
	session_options.enable_mem_pattern = "DmlExecutionProvider" not in ORT_ACCELERATE_PROVIDERS
	for key, value in {
		"session.set_denormal_as_zero": "1",
		"session.intra_op.allow_spinning": "1",
		"session.inter_op.allow_spinning": "1",
		"session.enable_quant_qdq_cleanup": "1",
		"session.qdq_matmulnbits_accuracy_level": "2" if ORT_FP16 else "4",
		"session.use_device_allocator_for_initializers": "1",
		"session.graph_optimizations_loop_level": "2",
		"optimization.enable_gelu_approximation": "1",
		"optimization.minimal_build_optimizations": "",
		"optimization.enable_cast_chain_elimination": "1",
		"optimization.disable_specified_optimizers": (
			"CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer"
			if ORT_FP16
			else ""
		),
	}.items():
		session_options.add_session_config_entry(key, value)
	run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
	disabled_optimizers = (
		["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
		if ORT_FP16
		else None
	)
	return session_options, run_options, disabled_optimizers


PROVIDERS = ORT_ACCELERATE_PROVIDERS or ["CPUExecutionProvider"]
PROVIDER_OPTIONS, DEVICE_TYPE = configure_provider()


def empty_ortvalue(
	shape: tuple[int, ...],
	dtype,
	device_type: str,
	device_id: int,
):
	return onnxruntime.OrtValue.ortvalue_from_shape_and_type(
		shape,
		dtype,
		device_type,
		device_id,
	)


def ortvalue_from_array(array: np.ndarray, device_type: str, device_id: int):
	return onnxruntime.OrtValue.ortvalue_from_numpy(array, device_type, device_id)


@dataclass
class DurationBuffers:
	token_array: np.ndarray
	token_value: object
	priors: object
	durations_array: np.ndarray
	durations: object
	frame_ends: np.ndarray
	binding: object


@dataclass
class DecodeBuffers:
	frame_index_array: np.ndarray
	frame_index_value: object
	binding: object


class InflectONNX:
	"""Torch-free Inflect v2 synthesis with shape-keyed reusable ORT buffers."""

	def __init__(
		self,
		model_dir: str | Path = ONNX_FOLDER,
		*,
		seed: int = RANDOM_SEED,
	) -> None:
		self.root = Path(model_dir).expanduser().resolve()
		self.metadata = read_metadata(self.root / METADATA_MODEL_NAME)
		self.seed = int(seed)
		self.model_sample_rate = int(
			require_metadata(self.metadata, "model_sample_rate")
		)
		self.sample_rate = int(require_metadata(self.metadata, "out_sample_rate"))
		self.fade_samples = int(require_metadata(self.metadata, "fade_samples"))
		self.add_blank = require_metadata(self.metadata, "add_blank") == "1"
		self.symbols = json.loads(require_metadata(self.metadata, "symbols_json"))
		self.symbol_to_id = {symbol: index for index, symbol in enumerate(self.symbols)}

		duration_path = self.root / require_metadata(
			self.metadata,
			"duration_model_file",
		)
		decode_path = self.root / require_metadata(self.metadata, "decode_model_file")
		shared_path = self.root / require_metadata(
			self.metadata,
			"shared_initializer_model_file",
		)
		self.provider = PROVIDERS[0]
		self.device_type = DEVICE_TYPE
		self.device_id = DEVICE_ID
		self._copy_inputs_to_device = self.device_type != "cpu"
		self.session_options, self.run_options, disabled_optimizers = (
			build_session_options()
		)
		shared_started = time.perf_counter()
		print_progress("Attaching shared ONNX initializers...")
		self._shared_arrays, self._shared_values = attach_shared_initializers(
			self.session_options,
			shared_path,
		)
		print_progress(
			f"Shared ONNX initializers ready in "
			f"{time.perf_counter() - shared_started:.2f}s."
		)
		session_started = time.perf_counter()
		print_progress(f"Loading ONNX graph 1/2: {duration_path.name}")
		self.duration_session = onnxruntime.InferenceSession(
			str(duration_path),
			sess_options=self.session_options,
			providers=PROVIDERS,
			provider_options=PROVIDER_OPTIONS,
			disabled_optimizers=disabled_optimizers,
		)
		print_progress(f"Loading ONNX graph 2/2: {decode_path.name}")
		with _ORT_SEED_LOCK:
			onnxruntime.set_seed(self.seed)
			self.decode_session = onnxruntime.InferenceSession(
				str(decode_path),
				sess_options=self.session_options,
				providers=PROVIDERS,
				provider_options=PROVIDER_OPTIONS,
				disabled_optimizers=disabled_optimizers,
			)
		print_progress(
			f"ONNX Runtime sessions ready in "
			f"{time.perf_counter() - session_started:.2f}s."
		)
		self._speed_array = np.asarray(1.0, dtype=self.speed_dtype)
		self._speed_value = ortvalue_from_array(
			self._speed_array,
			self.device_type,
			self.device_id,
		)
		self._variation_array = np.asarray(0.667, dtype=self.variation_dtype)
		self._variation_value = ortvalue_from_array(
			self._variation_array,
			self.device_type,
			self.device_id,
		)
		self._duration_cache: dict[int, DurationBuffers] = {}
		self._decode_cache: dict[tuple[int, int], DecodeBuffers] = {}
		self._boundary_pause_samples = {
			ending: round(self.sample_rate * seconds)
			for ending, seconds in BOUNDARY_PAUSE_SECONDS.items()
		}
		self._default_pause_samples = round(
			self.sample_rate * DEFAULT_BOUNDARY_PAUSE_SECONDS
		)
		self._lock = threading.RLock()

		self._run_frontend = frontend_phonemes

	def _prepare_duration_buffers(self, text: str) -> DurationBuffers:
		phonemes = self._run_frontend(text)
		try:
			phoneme_ids = [self.symbol_to_id[symbol] for symbol in phonemes]
		except KeyError as exc:
			raise ValueError(f"Unsupported frontend phoneme: {exc.args[0]!r}.") from exc
		token_count = len(phoneme_ids) * 2 + 1 if self.add_blank else len(phoneme_ids)
		buffers = self._duration_buffers(token_count)
		if self.add_blank:
			buffers.token_array.fill(0)
			buffers.token_array[1::2] = phoneme_ids
		else:
			buffers.token_array[...] = phoneme_ids
		if self._copy_inputs_to_device:
			buffers.token_value.update_inplace(buffers.token_array)
		return buffers

	def _duration_buffers(self, token_count: int) -> DurationBuffers:
		buffers = self._duration_cache.get(token_count)
		if buffers is not None:
			return buffers
		token_array = np.empty(token_count, dtype=self.token_dtype)
		token_value = ortvalue_from_array(
			token_array,
			self.device_type,
			self.device_id,
		)
		priors = empty_ortvalue(
			(token_count, self.priors_width),
			self.priors_dtype,
			self.device_type,
			self.device_id,
		)
		durations_array = np.empty(token_count, dtype=self.duration_dtype)
		durations = ortvalue_from_array(durations_array, "cpu", 0)
		frame_ends = np.empty(token_count, dtype=self.duration_dtype)
		binding = self.duration_session.io_binding()
		binding.bind_ortvalue_input(self.duration_token_input_name, token_value)
		binding.bind_ortvalue_input(self.duration_speed_input_name, self._speed_value)
		if self.device_type == "cuda":
			binding.bind_output(
				self.duration_priors_output_name,
				self.device_type,
				self.device_id,
			)
		else:
			binding.bind_ortvalue_output(self.duration_priors_output_name, priors)
		binding.bind_ortvalue_output(self.duration_values_output_name, durations)
		buffers = DurationBuffers(
			token_array,
			token_value,
			priors,
			durations_array,
			durations,
			frame_ends,
			binding,
		)
		self._duration_cache[token_count] = buffers
		return buffers

	def _decode_buffers(
		self,
		token_count: int,
		frame_count: int,
		priors,
	) -> DecodeBuffers:
		key = (token_count, frame_count)
		buffers = self._decode_cache.get(key)
		if buffers is not None:
			return buffers
		frame_index_array = np.empty(frame_count, dtype=self.frame_index_dtype)
		frame_index_value = ortvalue_from_array(
			frame_index_array,
			self.device_type,
			self.device_id,
		)
		binding = self.decode_session.io_binding()
		binding.bind_ortvalue_input(self.decode_priors_input_name, priors)
		binding.bind_ortvalue_input(
			self.decode_frame_index_input_name,
			frame_index_value,
		)
		binding.bind_ortvalue_input(
			self.decode_variation_input_name,
			self._variation_value,
		)
		binding.bind_output(self.decode_waveform_output_name, "cpu", 0)
		buffers = DecodeBuffers(
			frame_index_array,
			frame_index_value,
			binding,
		)
		self._decode_cache[key] = buffers
		return buffers

	@staticmethod
	def _fill_frame_indices(destination: np.ndarray, frame_ends: np.ndarray) -> None:
		frame_count = int(frame_ends[-1])
		destination.fill(0)
		destination[frame_ends[:-1]] = 1
		np.cumsum(destination, dtype=destination.dtype, out=destination)

	def _synthesize_tokens(
		self,
		duration_buffers: DurationBuffers,
	) -> np.ndarray:
		token_count = int(duration_buffers.token_array.size)
		if self.device_type == "cuda":
			duration_buffers.binding.bind_output(
				self.duration_priors_output_name,
				self.device_type,
				self.device_id,
			)
		self.duration_session.run_with_iobinding(
			duration_buffers.binding,
			run_options=self.run_options,
		)
		if self.device_type == "cuda":
			duration_buffers.priors.update_inplace(
				duration_buffers.binding.get_outputs()[0]
			)
		durations = duration_buffers.durations_array
		np.cumsum(
			durations,
			dtype=np.int32,
			out=duration_buffers.frame_ends,
		)
		frame_count = int(duration_buffers.frame_ends[-1])
		decode_buffers = self._decode_buffers(
			token_count,
			frame_count,
			duration_buffers.priors,
		)
		self._fill_frame_indices(
			decode_buffers.frame_index_array,
			duration_buffers.frame_ends,
		)
		if self._copy_inputs_to_device:
			decode_buffers.frame_index_value.update_inplace(
				decode_buffers.frame_index_array
			)
		self.decode_session.run_with_iobinding(
			decode_buffers.binding,
			run_options=self.run_options,
		)
		waveform, = decode_buffers.binding.copy_outputs_to_cpu()
		return waveform

	def synthesize(
		self,
		text: str,
		*,
		speed: float = 1.0,
		variation: float = 0.667,
	) -> tuple[int, np.ndarray]:
		normalized = " ".join(text.split())
		with self._lock:
			self._speed_array[...] = self.speed_dtype.type(speed)
			self._variation_array[...] = self.variation_dtype.type(variation)
			if self._copy_inputs_to_device:
				self._speed_value.update_inplace(self._speed_array)
				self._variation_value.update_inplace(self._variation_array)
			chunks = split_text(normalized)
			pieces: list[np.ndarray] = []
			pauses: list[int] = []
			for index, chunk in enumerate(chunks):
				duration_buffers = self._prepare_duration_buffers(chunk)
				waveform = self._synthesize_tokens(
					duration_buffers,
				)
				pieces.append(waveform)
				if index + 1 < len(chunks):
					ending = chunk.rstrip()[-1:] if chunk.strip() else ""
					pauses.append(
						self._boundary_pause_samples.get(
							ending,
							self._default_pause_samples,
						)
					)

			if len(pieces) == 1:
				return self.sample_rate, pieces[0]
			total_samples = sum(piece.size for piece in pieces) + sum(pauses)
			output = np.zeros(total_samples, dtype=self.waveform_dtype)
			offset = 0
			for index, piece in enumerate(pieces):
				output[offset : offset + piece.size] = piece
				offset += piece.size
				if index < len(pauses):
					offset += pauses[index]
			return self.sample_rate, output

	def save(
		self,
		text: str,
		output: str | Path,
		**synthesis_options: object,
	) -> Path:
		destination = Path(output).expanduser()
		destination.parent.mkdir(parents=True, exist_ok=True)
		sample_rate, waveform = self.synthesize(text, **synthesis_options)
		write_waveform = (
			waveform.astype(np.float32)
			if waveform.dtype == np.float16
			else waveform
		)
		subtype = "PCM_16" if np.issubdtype(waveform.dtype, np.integer) else "FLOAT"
		sf.write(destination, write_waveform, sample_rate, subtype=subtype)
		return destination

pipeline_started = time.perf_counter()
print_progress(f"Reading package metadata: {ONNX_FOLDER / METADATA_MODEL_NAME}")
engine = InflectONNX(ONNX_FOLDER, seed=RANDOM_SEED)
initialized = time.perf_counter()
print_progress("Synthesizing target text...")
destination = engine.save(
	TARGET_TEXT,
	GENERATED_AUDIO_PATH,
	speed=SPEED,
	variation=VARIATION,
)
finished = time.perf_counter()
info = sf.info(destination)
audio_seconds = info.frames / info.samplerate
synthesis_seconds = finished - initialized
print(
	f"Generate complete. Saved {destination} ({audio_seconds:.2f}s audio, "
	f"{engine.sample_rate} Hz) with {engine.provider}; "
	f"init={initialized - pipeline_started:.2f}s, "
	f"inference={synthesis_seconds:.2f}s, "
	f"RTF={synthesis_seconds / max(audio_seconds, 1e-9):.3f}.",
	flush=True,
)
print_progress(f"Pipeline complete in {finished - pipeline_started:.2f}s.")
