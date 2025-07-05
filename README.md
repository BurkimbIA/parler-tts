## Table of Contents

1. [Introduction](#introduction)
2. [Our Datasets](#our-datasets)
3. [Working with Audio Datasets](#working-with-audio-datasets)
4. [Transformers for Audio](#transformers-for-audio)
5. [Saving Steps to Drive Path](#saving-steps-to-drive-path)
6. [References & Further Reading](#references--further-reading)

# Introduction

Toolskit-TTS is a comprehensive toolkit designed to facilitate research and development in Text-to-Speech (TTS) systems. It provides resources, scripts, and best practices for working with TTS datasets, training models, and evaluating results. The toolkit supports multilingual and multi-speaker scenarios, making it suitable for a wide range of TTS research projects.

# Our Datasets

Our current dataset consists of approximately 50 hours of audio sourced from Bible JW and Mooreburkina. These audio files have undergone preprocessing steps such as denoising and enhancement. For more details, visit:

`s3://burkimbia/audios/final_dataset`

# Working with Audio Datasets

Working with audio datasets for TTS involves several challenges, including alignment, diversity, and quality. For a detailed guide on audio preprocessing and dataset creation, refer to this blog:

[Audio Dataset Creation Guide](https://sawallesalfo.github.io/blog/2025/03/23/traitement-des-audios-pour-la-cr%C3%A9ation-de-datasets-audio/)

# Transformers for Audio

Transformers have revolutionized the field of audio processing, enabling advanced capabilities in TTS. Key resources for understanding and implementing audio transformers include:

- [Audio Transformers Repository](https://github.com/anyantudre/Audio-Transformers-Hugging-Face/tree/main)
- [Hugging Face Audio Course](https://huggingface.co/learn/audio-course)

These resources provide a comprehensive overview of transformer architectures and their applications in audio tasks.

# Saving Steps to Drive Path

To save outputs or intermediate steps to a specific drive path, follow these instructions:

1. **Specify the Drive Path**
   - Ensure the drive path is accessible and has sufficient storage.
   - Example: `D:\TTS_Outputs\`

2. **Modify Scripts**
   - Update the output directory in relevant scripts or configuration files.
   - Example in Python:
     ```python
     output_path = "D:\TTS_Outputs\"
     save_to_path(output_path, data)
     ```

3. **Automate Saving**
   - Use logging or checkpointing mechanisms to save intermediate results.
   - Example:
     ```python
     def save_checkpoint(model, path):
         torch.save(model.state_dict(), path)
     save_checkpoint(model, "D:\TTS_Outputs\checkpoint.pt")
     ```

4. **Verify Saved Files**
   - Check the drive path to ensure files are saved correctly.
   - Use tools like `os.listdir()` to list saved files.

# References & Further Reading

Explore the following references for additional insights and tools:

- [Coqui TTS](https://coqui.ai/)
- [Hugging Face Parler-TTS](https://huggingface.co/parler-tts)
- [Coqui-AI GitHub Repository](https://github.com/coqui-ai/TTS)