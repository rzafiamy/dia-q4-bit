<p align="center">
  <a href="https://github.com/nari-labs/dia">
    <img src="./dia/static/images/banner.png" alt="Dia Banner">
  </a>
</p>

<p align="center">
  <a href="https://tally.so/r/meokbo" target="_blank">
    <img alt="Join Waitlist" src="https://img.shields.io/badge/Join-Waitlist-white?style=for-the-badge">
  </a>
  <a href="https://discord.gg/pgdB5YRe" target="_blank">
    <img src="https://img.shields.io/badge/Discord-Join%20Chat-7289DA?logo=discord&style=for-the-badge">
  </a>
  <a href="https://github.com/nari-labs/dia/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge" alt="LICENSE">
  </a>
</p>

<p align="center">
  <a href="https://huggingface.co/nari-labs/Dia-1.6B">
    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg-dark.svg" alt="Model on Hugging Face" height="42">
  </a>
  <a href="https://huggingface.co/spaces/nari-labs/Dia-1.6B">
    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg-dark.svg" alt="Demo Space on Hugging Face" height="38">
  </a>
</p>

---

# 🎙️ Dia - Text to Speech Dialogue Model

Dia is a 1.6B parameter text-to-speech model developed by **Nari Labs**, designed to **generate highly realistic dialogues directly from transcripts**. It supports emotional conditioning and non-verbal cues such as laughter, coughing, and more.

> 🧪 Pretrained model checkpoints and inference code are publicly available to accelerate research and experimentation.  
> 💬 English only (for now).  
> 🔬 Demo available on Hugging Face Spaces.

---

## 🔧 About This Fork

This is a **community-enhanced fork** of the original [nari-labs/dia](https://github.com/nari-labs/dia) repository. It focuses on **performance optimizations and accessibility for personal hardware**.

### 🚀 Enhancements in This Fork:

- 🧠 **Quantization (4-bit & INT8)**  
  Reduces VRAM usage by nearly 50%. The model now runs on **GPUs with just 8GB VRAM**, consuming **less than 6GB** in practice.

- ⚡ **Flash Attention Integration**  
  Experimental support for [Flash Attention](https://github.com/Dao-AILab/flash-attention) to speed up inference and reduce memory footprint.

- 🔄 **Continued Development**  
  Ongoing efforts to improve inference speed, memory efficiency, and model accessibility.

---

## ✨ Features

- 🔈 Generate expressive dialogue using `[S1]`, `[S2]` tags.
- 🤖 Realistic non-verbal sounds: `(laughs)`, `(sighs)`, `(coughs)`, etc.
- 🧬 Optional **voice cloning** via reference audio.
- 🪄 Supports speaker conditioning and output diversity.
- 💡 Simple Python API for generation and saving audio.

---

## ⚡ Quickstart

### 1. Install via pip (from this fork)

```bash
pip install git+https://github.com/<your-username>/dia.git
```

### 2. Launch the Gradio UI

```bash
git clone https://github.com/<your-username>/dia.git
cd dia
uv run app.py
```

Or using a Python virtual environment:

```bash
git clone https://github.com/<your-username>/dia.git
cd dia
python -m venv .venv
source .venv/bin/activate
pip install -e .
python app.py
```

> ℹ️ Voices vary between runs unless you condition output with an audio prompt or set a fixed seed.

---

## 🐍 Usage in Python

```python
from dia.model import Dia

model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on GitHub or Hugging Face."

output = model.generate(text, use_torch_compile=True, verbose=True)
model.save_audio("simple.mp3", output)
```

> 📢 CLI and PyPI package coming soon.

---

## 🖥️ Hardware & Inference Performance

Tested on PyTorch 2.0+ and CUDA 12.6. CPU support coming soon.

| Precision   | Real-time (w/ compile) | Real-time (w/o compile) | VRAM Usage |
|-------------|------------------------|--------------------------|------------|
| `float16`   | x2.2                   | x1.3                     | ~10 GB     |
| `bfloat16`  | x2.1                   | x1.5                     | ~10 GB     |
| `float32`   | x1.0                   | x0.9                     | ~13 GB     |
| `int8/4bit` | ✅ Efficient (forked)   | ✅ Efficient (forked)    | **< 6 GB** |

---

## 🪪 License

Apache License 2.0 – see the [LICENSE](LICENSE) file for full details.

---

## ⚠️ Disclaimer

This model is for **research and educational use only**. By using it, you agree **not** to:

- Mimic real identities without consent.
- Generate misleading or harmful content.
- Use it for illegal or malicious purposes.

---

## 🔭 TODO / Roadmap

- [x] 4-bit / INT8 quantization support.
- [x] Flash Attention integration.
- [ ] CPU inference support.
- [ ] Docker support (incl. ARM/MacOS).
- [ ] Public PyPI release & CLI.
- [ ] Larger model versions.

---

## 🤝 Contributing

This fork is maintained by the open-source community.  
PRs are welcome! Join us on [Discord](https://discord.gg/pgdB5YRe) to collaborate or share ideas.

---

## 🙏 Acknowledgements

- [Nari Labs](https://github.com/nari-labs) for the original Dia model.
- [Google TPU Research Cloud](https://sites.research.google/trc/about/) for computing resources.
- Research inspiration from: [SoundStorm](https://arxiv.org/abs/2305.09636), [Parakeet](https://jordandarefsky.com/blog/2024/parakeet/), and [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).
- [Hugging Face](https://huggingface.co) for hosting weights and demo space.

---

## ⭐ Star History

<a href="https://www.star-history.com/#nari-labs/dia&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
  </picture>
</a>