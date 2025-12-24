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

### Install Ollama

1. Install **Ollama** on the host system following the official instructions:
   https://ollama.com

2. Pull a recommended model (CPU-friendly and well-supported):

```bash
ollama pull llama3.2
```

3. Make sure Ollama is running and reachable by OpenWebUI.
   The default Ollama endpoint is:

```
http://ollama:11434
```

4. Configure OpenWebUI (or docker-compose) to use this endpoint as the LLM backend.

---

### Image Generation Configuration

To enable **image generation** through OpenWebUI using the AUTOMATIC1111-compatible backend, configure the following environment variables:

```
ENABLE_IMAGE_GENERATION=true
IMAGE_GENERATION_ENGINE=automatic1111
IMAGE_GENERATION_MODEL=<your_model>
IMAGE_SIZE=512x512
IMAGE_STEPS=4
AUTOMATIC1111_BASE_URL=http://fast-a1111:7860
AUTOMATIC1111_API_AUTH=none
```

Once configured, OpenWebUI will expose the **image generation** feature, allowing you to generate images directly from text prompts using the built-in image generation function.

---

### Text-to-Speech (TTS) Configuration

To enable **text-to-speech** support via the Chatterbox backend, configure the following environment variables:

```
AUDIO_TTS_ENGINE=openai
AUDIO_TTS_OPENAI_API_BASE_URL=http://fast-chatterbox:8000/v1
AUDIO_TTS_OPENAI_API_KEY=none
```

After configuration, OpenWebUI will allow audio generation from text responses using the integrated TTS features.

#### Voice Cloning

TODO:
- Add documentation for zero-shot voice cloning
- Explain voice samples ingestion and selection workflow


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
