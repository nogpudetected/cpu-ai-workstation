# app.py
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path

import io
import os
import torch
import torchaudio as ta
import soundfile as sf
import logging
import shutil
import uuid

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

app = FastAPI(title="fast-chatterbox")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class Settings:
    OPENWEBUI_UPLOADS_DIR: Path = Path(os.getenv("OPENWEBUI_UPLOADS_DIR", "/app/uploads"))
    SPEAKERS_DIR: Path = Path(os.getenv("SPEAKERS_DIR", "/app/speakers"))
    
    DEVICE: str = os.getenv("DEVICE", "cpu")
    
    DEFAULT_CFG_SCALE: float = 1.0
    DEFAULT_EXAGGERATION: float = 0.5
    DEFAULT_SAMPLE_RATE: int = 24000
    DEFAULT_LANG: str = "it"

settings = Settings()

# ==========================================
# MONKEY PATCH: FIX FORZATURA DEVICE
# ==========================================
# Salva la funzione originale di torch
_original_torch_load = torch.load
def _force_device_load(*args, **kwargs):
    # Forza il caricamento sul device configurato indipendentemente da come è stato salvato il file
    kwargs['map_location'] = torch.device(settings.DEVICE)
    # Richiama la funzione originale con i nuovi parametri
    return _original_torch_load(*args, **kwargs)
# Sovrascive la funzione load dentro il modulo torch
torch.load = _force_device_load
print("--> Patch applicata: torch.load forzerà map_location=settings.DEVICE")
# ==========================================
# Ora quando chiamerà torch.load(), userà la versione "truccata".

tts_model = ChatterboxMultilingualTTS.from_pretrained(device=settings.DEVICE)
sample_rate = tts_model.sr

# Opzionale: API Key
API_KEY = os.getenv("TTS_API_KEY", None)

class TTSRequest(BaseModel):
    input: str
    model: str
    voice: str
    # parametri opzionali
    cfg_weight: float = settings.DEFAULT_CFG_SCALE
    exaggeration: float = settings.DEFAULT_EXAGGERATION
    seed: int | None = None
    language_id: str = settings.DEFAULT_LANG

class CloneVoiceRequest(BaseModel):
    wav_path: str

@app.post("/v1/audio/speech")
def tts(req: TTSRequest, authorization: str | None = Header(None)):
    
    logging.info(f"start /v1/audio/speech - {req}")

    #if API_KEY:
    #    if not authorization or not authorization.startswith("Bearer "):
    #        raise HTTPException(status_code=401, detail="Missing Authorization")
    #    token = authorization.split(" ", 1)[1]
    #    if token != API_KEY:
    #        raise HTTPException(status_code=403, detail="Invalid API Key")

    try:
        if req.voice != "default":
            logging.info("Starting zero-shot voice cloning...")
            try:
                wav = tts_model.generate(
                    req.input,
                    cfg_weight=req.cfg_weight,
                    exaggeration=req.exaggeration,
                    # seed=req.seed, # Se serve
                    language_id=req.language_id,
                    audio_prompt_path=settings.SPEAKERS_DIR / f"{req.voice}.wav"
                )
                ta.save("/tmp/valettoviastreamnondaunpath.wav", wav, tts_model.sr)
                return FileResponse("/tmp/valettoviastreamnondaunpath.wav", media_type="audio/wav", filename="output.wav")
            except Exception as e:
                logging.info(f"Exception caught during generation: {e}")
                raise HTTPException(status_code=500, detail=f"TTS clone error: {e}")
        else:
            logging.info("Starting Standard TTS...")
            try:
                wav_tensor = tts_model.generate(
                    req.input,
                    cfg_weight=req.cfg_weight,
                    exaggeration=req.exaggeration,
                    # seed=req.seed, # Se serve
                    language_id=req.language_id
                )
                wav_numpy = wav_tensor.cpu().numpy()
                wav_numpy = wav_numpy.squeeze()
                buf = io.BytesIO()
                sample_rate = settings.DEFAULT_SAMPLE_RATE
                sf.write(buf, wav_numpy, sample_rate, format='WAV')
                buf.seek(0)
                return StreamingResponse(buf, media_type="audio/wav")
            except Exception as e:
                logging.info(f"Exception caught: {e}")
                raise HTTPException(status_code=500, detail=f"TTS error: {e}")
    except Exception as e:
        logging.info(f"Exception caught: {e}")
        raise HTTPException(status_code=500, detail=f"TTS clone error: {e}")

@app.post("/store-voice")
async def clone_voice(request: CloneVoiceRequest):
    """
    Riceve un file WAV, lo salva come nuovo speaker e restituisce il nome.
    """
    logger.info(f"Start /store-voice... {request}")
    try:
        source_wav_name = request.wav_path.split("/")[-1]
        source_file = settings.OPENWEBUI_UPLOADS_DIR / source_wav_name
        if not os.path.exists(source_file):
            logger.error(f"File non trovato nel percorso: {source_file}")
            raise HTTPException(status_code=404, detail="File sorgente non trovato")
        clone_speaker_name = f"speaker_{uuid.uuid4().hex[:8]}"
        output_file = settings.SPEAKERS_DIR / f"{clone_speaker_name}.wav"
        shutil.copy2(source_file, output_file)
        logging.info("finished /store-voice")
        return {"speaker": clone_speaker_name}

    except Exception as e:
        logger.error(f"Errore durante la clonazione: {e}")
        raise HTTPException(status_code=500, detail=str(e))

