# Text-to-Speech-TTS-ONNX
   Utilizes ONNX Runtime for TTS model. 
### Features  
1. **Supported Models**:
   - [KaniTTS](https://github.com/nineninesix-ai/kani-tts)
   - [F5-TTS](https://github.com/SWivid/F5-TTS)
   - [IndexTTS](https://github.com/index-tts/index-tts)
   - [BigVGAN](https://github.com/NVIDIA/BigVGAN) (It is part of the TTS module.)
   - [VoxCPM](https://www.modelscope.cn/models/OpenBMB/VoxCPM1.5)

3. **End-to-End Processing**:  
   - The solution includes internal `STFT/ISTFT` processing.  
   - Input: `reference audio` + `text`  
   - Output: `generated speech`  

4. **Optimize**:  
   - The key components enable 100% deployment of GPU operators. 

5. **Resources**:  
   - [Explore More Projects](https://github.com/DakeQQ?tab=repositories)  

---

### 性能 Performance  
| OS           | Device       | Backend           | Model               | Time Cost in Seconds <br> (reference audio: 6s / generates approximately 15 words of speech) | RTF |
|:------------:|:------------:|:-----------------:|:-------------------:|:-------------------------------------------------------------------------:|:----|
| Ubuntu-24.04 | Laptop       | CPU <br> i7-1165G7 | F5-TTS<br>F32      |        180 <br> (NFE=32)                                                  | 60 |
| Ubuntu-24.04 | Laptop       | GPU <br> MX150     | F5-TTS<br>F32      |        62 <br> (NFE=32)                                                   | 21 |
| Ubuntu-24.04 | Laptop       | CPU <br> i7-1165G7 | IndexTTS<br>F32    |        18                                                                 | 6 |
| Ubuntu-24.04 | Laptop       | GPU <br> MX150     | BigVGAN V2 24khz_100band_256x <br>F16    |        4.6 <br> input mel = (1, 100, 512)           | 1.53 |
| Ubuntu-24.04 | Laptop       | CPU <br> i7-1165G7 | KaniTTS<br>Q8F32   |        4.2                                                                | 1.4 |
| Ubuntu-24.04 | Laptop       | CPU <br> i7-1165G7 | KaniTTS<br>Q4F32   |        2.6                                                                | 0.87 |
| Ubuntu-24.04 | Desktop      | CPU <br> i3-12300  | VoxCPM-1.5<br>Q8F32   |        9                                                                   | 1.5 |
| Ubuntu-24.04 | Desktop      | GPU <br> 5060Ti    | VoxCPM-1.5<br>F16     |        1.2                                                                 | 0.2 |


---

### To-Do List  
- [ ] Beam Search
---

# Text-to-Speech-TTS-ONNX
通过 ONNX Runtime 实现运行 TTS 模型。

### 功能  
1. **支持的模型**：
   - [KaniTTS](https://github.com/nineninesix-ai/kani-tts)
   - [F5-TTS](https://github.com/SWivid/F5-TTS)
   - [IndexTTS](https://github.com/index-tts/index-tts)
   - [BigVGAN](https://github.com/NVIDIA/BigVGAN) (它是TTS模块的一部分)
   - [VoxCPM](https://www.modelscope.cn/models/OpenBMB/VoxCPM1.5)

3. **端到端处理**：  
   - 解决方案内置 `STFT/ISTFT` 处理。  
   - 输入：`参考音频` + `文本`  
   - 输出：`生成的语音`
     
4. **优化**:  
   - 模型关键组件实现了 100% GPU 算子部署。
     
5. **资源**：  
   - [探索更多项目](https://github.com/DakeQQ?tab=repositories)  
---
