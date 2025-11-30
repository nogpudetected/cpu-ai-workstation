# CPU-Optimized Local AI Stack

### Ollama + OpenWebUI with Custom CPU Backends for Stable Diffusion & TTS

This repository provides a fully local **CPU-only AI environment** combining:

* **Ollama** for LLM inference
* **OpenWebUI** as the unified frontend
* **Custom Python backends** for:

  * Stable Diffusion (CPU-only image generation)
  * Chatterbox TTS (offline speech synthesis)

The stack is designed for systems without GPUs: homelabs, mini-PCs, servers, and privacy-focused offline setups.

---

## ✨ Features

* **LLMs via Ollama** — fully local and CPU-optimized
* **Stable Diffusion CPU backend** — optimized inference pipeline
* **Chatterbox TTS backend** — lightweight text-to-speech
* **OpenWebUI integration** — UI support for LLM, SD, and TTS
* **Modular architecture** — each backend runs independently
* **Zero GPU required**

---

## 📁 Repository Structure

```
repo/
├── backends/
│   ├── sd_cpu_backend/
│   │   ├── sd_backend.py
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── tts_chatterbox_backend/
│       ├── chatterbox_backend.py
│       ├── requirements.txt
│       └── README.md
├── openwebui/
│   ├── config_example.json
│   └── backends_registration.md
├── ollama/
│   └── models.md
├── examples/
│   ├── image_generation_example.py
│   ├── tts_example.py
│   └── prompts.md
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install backend dependencies

**Stable Diffusion (CPU)**

```bash
cd backends/sd_cpu_backend
pip install -r requirements.txt
```

**Chatterbox TTS**

```bash
cd ../tts_chatterbox_backend
pip install -r requirements.txt
```

### 3. Start the backends

**Start SD backend**

```bash
python sd_backend.py
```

**Start TTS backend**

```bash
python chatterbox_backend.py
```

### 4. Configure OpenWebUI

```bash
cp openwebui/config_example.json ~/.config/openwebui/config.json
```

### 5. (Optional) Install Ollama models

```bash
ollama pull llama3
ollama pull qwen2
```

---

## 🧪 Usage Examples

### Image Generation

```python
from sd_backend import generate_image

img = generate_image(
    prompt="Watercolor painting of a futuristic city",
    steps=20
)
img.save("result.png")
```

### Text-to-Speech

```python
from chatterbox_backend import synthesize

audio = synthesize("Hello! This audio was generated on CPU.")
with open("speech.wav", "wb") as f:
    f.write(audio)
```

---

## ⚙️ Configuration

### Environment Variables

```
SD_MODEL_PATH=./models/sd/
TTS_MODEL_PATH=./models/chatterbox/
BACKEND_PORT_SD=5001
BACKEND_PORT_TTS=5002
```

### OpenWebUI backend registration

```json
"custom_backends": [
    { "name": "sd_cpu", "url": "http://localhost:5001" },
    { "name": "chatterbox_tts", "url": "http://localhost:5002" }
]
```

---

## 🧱 Architecture

```
OpenWebUI
   │
   ├── Ollama (LLMs)
   ├── SD CPU Backend (Stable Diffusion)
   └── Chatterbox TTS Backend
```

All components run fully locally and **it is not needed any GPU**.

---

## 🧭 Roadmap

* [ ] CPU-only Docker support
* [ ] Unified installation script
* [ ] Quantized SD pipeline
* [ ] Whisper CPU backend
* [ ] Benchmark suite

---

## 🤝 Contributing

Pull requests, issues, and suggestions are welcome.

---

## 📝 License

Released under the **MIT License**. See `LICENSE` for details.
