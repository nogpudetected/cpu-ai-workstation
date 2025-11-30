# CPU-Optimized Local AI Stack

### Ollama + OpenWebUI with Custom CPU Backends for Stable Diffusion & TTS

This repository provides a fully local **CPU-only AI environment** combining:

* **Ollama** for LLM inference
* **OpenWebUI** as the unified frontend
* **A custom Python backend** for Stable Diffusion implementation (simulates **AUTOMATIC1111** APIs)
  * txt2img
  * img2img
  * LCM optimization with LoRA
* **A custom Python backend** for Chatterbox TTS implementation
  * offline speech synthesis
  * zero-shot voice cloning

The stack is designed for systems without GPUs: homelabs, mini-PCs, servers, and privacy-focused offline setups.

---

## ✨ Features

* **LLMs via Ollama** — fully local and CPU-optimized by models quantization
* **Stable Diffusion CPU backend** — optimized inference pipeline for images generation
* **Chatterbox TTS backend** — lightweight offline text-to-speech and zero-shot voice cloning
* **OpenWebUI integration** — UI support for LLM, SD, TTS and RAG
* **Modular architecture** — each backend runs independently
* **Zero GPU required**

---

## 📁 Repository Structure

```
repo/
├── fast-a1111/
│   ├── Dockerfile
│   └── server.py
├── fast-chatterbox/
│   ├── Dockerfile
│   └── server.py
├── ollama/
│   └── .gitkeep
├── openwebui/
│   └── extensions/
│       └── tools/
│           └── openwebui-voicecloner-tool.py
├── licenses/
│   └── LICENSE
├── docker-compose.yml
├── LICENSE
└── .gitignore
```

---

## 🛠 Installation

### Installation via Docker

```bash
docker compose up --build -d 
```

### Installation via Python

```bash
cd backends/sd_cpu_backend
pip install -r requirements.txt
```
## ⚙️ Configuration

### Install Ollama models
...

## 🧪 Usage Examples

### Image Generation

<screenshot>

### Text-to-Speech

<screenshot>

...
 
---

## ⚙️ Configuration

### Environment Variables

```
SD_MODEL_PATH=./models/sd/
TTS_MODEL_PATH=./models/chatterbox/
BACKEND_PORT_SD=5001
BACKEND_PORT_TTS=5002
```

## 🧭 Roadmap

* [X] CPU-only Docker support
* [X] Unified installation script
* [X] Quantized SD pipeline
* [ ] Whisper CPU backend
* [X] Benchmark suite

---

## 🤝 Contributing

Pull requests, issues, and suggestions are welcome.

---

## 📝 License

Released under the **MIT License**. See `LICENSE` for details.
