# Text-to-Speech-TTS-ONNX
   Utilizes ONNX Runtime for TTS model. 
### Features  
1. **Supported Models**:  
   - [IndexTTS](https://github.com/index-tts/index-tts)  
   - [F5-TTS](https://github.com/SWivid/F5-TTS)

2. **End-to-End Processing**:  
   - The solution includes internal `STFT/ISTFT` processing.  
   - Input: `reference audio` + `text`  
   - Output: `generated speech`  

3. **Optimize**:  
   - The key components enable 100% deployment of GPU operators. 

4. **Resources**:  
   - [Explore More Projects](https://github.com/DakeQQ?tab=repositories)  

---

### 性能 Performance  
| OS           | Device       | Backend           | Model        | Time Cost in Seconds <br> (reference audio: 6s / generates approximately 15 words of speech) |
|:------------:|:------------:|:-----------------:|:------------:|:-------------------------------------------------------------------------:|
| Ubuntu-24.04 | Laptop       | CPU <br> i7-1165G7 | F5-TTS      |        180 (NFE=32)                                                       |
| Ubuntu-24.04 | Laptop       | GPU <br> MX150     | F5-TTS      |        62 (NFE=32)                                                        |
| Ubuntu-24.04 | Laptop       | CPU <br> i7-1165G7 | IndexTTS    |        18                                                                 |

---

### To-Do List  
- [ ] Beam Search
---

## Audio-Denoiser-ONNX  
通过 ONNX Runtime 实现音频降噪，提升音质清晰度。

### 功能  
1. **支持的模型**：  
   - [IndexTTS](https://github.com/index-tts/index-tts)  
   - [F5-TTS](https://github.com/SWivid/F5-TTS)

2. **端到端处理**：  
   - 解决方案内置 `STFT/ISTFT` 处理。  
   - 输入：`参考音频` + `文本`  
   - 输出：`生成的语音`
     
3. **优化**:  
   - 模型关键组件实现了 100% GPU 算子部署。
     
4. **资源**：  
   - [探索更多项目](https://github.com/DakeQQ?tab=repositories)  
---
