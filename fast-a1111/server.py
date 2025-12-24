import os
import asyncio
import json
import gc
import logging
import io
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import torch
import aiosqlite
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from diffusers import StableDiffusionPipeline, LCMScheduler, AutoPipelineForText2Image

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
class Settings:
    MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", "/app/models"))
    LORA_DIR: Path = Path(os.getenv("LORA_DIR", "/app/models/lora"))
    DB_PATH: str = os.getenv("DB_PATH", "/app/data/fast_a1111.db")
    
    DEVICE: str = os.getenv("DEVICE", "cpu")
    # usiamo float32 per ottimizzare su CPU
    DTYPE: torch.dtype = torch.float32 
    
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT", "1"))

    # ottimizzazioni LCM (fondamentale per CPU)
    # se true, inietta LCM-LoRA in ogni modello caricato per ridurre step a 4-8
    USE_LCM_OPTIMIZATION: bool = os.getenv("USE_LCM", "true").lower() == "true"
    
    # limiti
    MAX_STEPS: int = int(os.getenv("MAX_STEPS", "8"))
    MAX_WIDTH: int = int(os.getenv("MAX_WIDTH", "768"))
    MAX_HEIGHT: int = int(os.getenv("MAX_HEIGHT", "768"))
    MAX_CFG_SCALE: float = 2.0

    DEFAULT_CFG_SCALE: float = 1.5

    # Modelli
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "v1-5-pruned-emaonly.safetensors")
    DEFAULT_LORA = os.getenv("DEFAULT_LORA", "lcm-lora-sdv1-5.safetensors")

settings = Settings()

# -----------------------------------------------------------
# 2. LOGGING SETUP
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# 3. GESTIONE DATABASE (Settings + Jobs)
# -----------------------------------------------------------
async def init_db():
    async with aiosqlite.connect(settings.DB_PATH) as db:
        # Tabella Jobs (Coda)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                params TEXT,
                model_name TEXT,
                status TEXT,
                result TEXT,
                error_msg TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Tabella Settings (Options API)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        # Default settings se vuoto
        await db.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", ("sd_model_checkpoint", f"{settings.DEFAULT_MODEL}"))
        await db.commit()

async def get_setting(key: str, default: str = "") -> str:
    async with aiosqlite.connect(settings.DB_PATH) as db:
        async with db.execute("SELECT value FROM settings WHERE key = ?", (key,)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else default

async def set_setting(key: str, value: str):
    async with aiosqlite.connect(settings.DB_PATH) as db:
        await db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
        await db.commit()

# -----------------------------------------------------------
# 4. MODEL MANAGER (Caricamento, LCM, LoRA)
# -----------------------------------------------------------
class ModelManager:
    def __init__(self):
        self.current_model_name: Optional[str] = None
        self.pipe = None
        self.lock = asyncio.Lock()

    def load_model(self, model_filename: str):
        # Se il modello è già in memoria, ritorna subito
        if self.current_model_name == model_filename and self.pipe is not None:
            return self.pipe

        logger.info(f"Richiesto cambio modello: {model_filename}")
        
        # Garbage Collection aggressiva
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            gc.collect()
            if settings.DEVICE == "cuda":
                torch.cuda.empty_cache()
            logger.info("Memoria liberata.")

        model_path = settings.MODEL_DIR / settings.DEFAULT_MODEL

        try:
            logger.info(f"Caricamento {model_filename} su {settings.DEVICE}...")
            
            self.pipe = StableDiffusionPipeline.from_single_file(
                str(model_path),
                dtype=settings.DTYPE,
                local_files_only=True, 
                use_safetensors=True,
                safety_checker=None, 
                requires_safety_checker=False
            )
            
            self.pipe.to(settings.DEVICE)
            
            if settings.USE_LCM_OPTIMIZATION:
                logger.info("Applicazione LCM LoRA...")
                try:
                    self.pipe.load_lora_weights(settings.LORA_DIR / settings.DEFAULT_LORA, adapter_name="lcm")
                    # Fuse loRA: fonde i pesi nel modello base per velocizzare l'inferenza su CPU
                    # (Operazione leggermente lenta all'avvio, ma velocizza la generazione)
                    self.pipe.fuse_lora() 
                    # Cambio scheduler in LCM
                    self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
                    logger.info("Modalità LCM Offline attivata (usa 4-8 steps).")
                except Exception as e:
                    logger.warning(f"LCM LoRA saltato (non trovato o errore): {e}")

            self.current_model_name = model_filename
            return self.pipe

        except Exception as e:
            logger.critical(f"CRITICAL: Errore caricamento modello: {e}")
            raise e

    def apply_lora(self, lora_names: List[str], weights: List[float]):
        """
        Logica placeholder per applicare LoRA aggiuntivi.
        Su diffusers richiede gestione attenta degli adapter.
        """
        if not self.pipe or not lora_names:
            return
        
        # Esempio semplificato: carica LoRA e fa merge
        # Per un sistema di produzione, meglio usare fuse_lora/unfuse_lora per evitare memory leaks
        pass 

model_manager = ModelManager()

# -----------------------------------------------------------
# 5. WORKER DI GENERAZIONE (Thread Safe)
# -----------------------------------------------------------
def process_generation_sync(job_id: int, model_name: str, params: Dict[str, Any]):
    try:
        logger.info(f"Job {job_id}: Inizio generazione con {model_name}")
        
        # Caricamento modello (sincrono ma dentro thread)
        pipe = model_manager.load_model(model_name)
        
        # Mapping parametri A1111 -> Diffusers
        generator = None
        if params.get("seed", -1) != -1:
            generator = torch.Generator(settings.DEVICE).manual_seed(params["seed"])
        
        # Esecuzione
        output = pipe(
            prompt=params.get("prompt"),
            negative_prompt=params.get("negative_prompt", ""),
            num_inference_steps=params.get("steps", settings.MAX_STEPS),
            width=params.get("width", settings.MAX_WIDTH),
            height=params.get("height", settings.MAX_HEIGHT),
            guidance_scale=params.get("cfg_scale", settings.DEFAULT_CFG_SCALE),
            generator=generator
        )
        
        image = output.images[0]
        
        # Converti in Base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        b64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return "completed", b64_img, None
    except Exception as e:
        logger.error(f"Job {job_id} Fallito: {e}")
        return "failed", None, str(e)

async def run_job_in_background(job_id: int, model_name: str, params: Dict[str, Any]):
    # 1. Update stato processing
    async with aiosqlite.connect(settings.DB_PATH) as db:
        await db.execute("UPDATE jobs SET status = 'processing' WHERE id = ?", (job_id,))
        await db.commit()

    # 2. Esecuzione (CPU Bound -> ThreadPool)
    async with model_manager.lock:
        loop = asyncio.get_running_loop()
        status, result, error = await loop.run_in_executor(
            None, process_generation_sync, job_id, model_name, params
        )

    # 3. Update stato finale
    async with aiosqlite.connect(settings.DB_PATH) as db:
        await db.execute(
            "UPDATE jobs SET status = ?, result = ?, error_msg = ? WHERE id = ?", 
            (status, result, error, job_id)
        )
        await db.commit()

# -----------------------------------------------------------
# 6. API & SCHEMI DI VALIDAZIONE
# -----------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    settings.LORA_DIR.mkdir(parents=True, exist_ok=True)
    await init_db()
    
    # Pre-load modello default (opzionale, rallenta avvio ma velocizza prima richiesta)
    default_model = await get_setting("sd_model_checkpoint", f"{settings.DEFAULT_MODEL}")
    logger.info(f"Startup: Modello selezionato {default_model}")
    
    yield
    
    # Shutdown
    if model_manager.pipe:
        del model_manager.pipe
        gc.collect()
    logger.info("Shutdown completato.")

app = FastAPI(title="fast-a1111", lifespan=lifespan)

# Global Exception Handler per robustezza
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__}
    )

# Schemi Pydantic con Validazione
class Txt2ImgRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt positivo")
    negative_prompt: Optional[str] = ""
    steps: int = Field(8, ge=1, le=settings.MAX_STEPS, description="Step di inferenza")
    width: int = Field(512, ge=64, le=settings.MAX_WIDTH)
    height: int = Field(512, ge=64, le=settings.MAX_HEIGHT)
    cfg_scale: float = Field(1.0, ge=0.0, le=settings.MAX_CFG_SCALE)
    seed: int = Field(-1)
    override_settings: Optional[Dict[str, Any]] = {}

class OptionsRequest(BaseModel):
    sd_model_checkpoint: Optional[str] = None

# -----------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------

@app.get("/sdapi/v1/sd-models")
async def list_models():
    """Ritorna la lista dei file fisici nella cartella models"""
    models = []
    # Cerca ricorsivamente o solo top level
    extensions = {".ckpt", ".safetensors"}
    files = [p for p in settings.MODEL_DIR.iterdir() if p.suffix in extensions]
    
    for file in files:
        models.append({
            "title": file.name,
            "model_name": file.name,
            "hash": "fakehash", # saltiamo calcolo hash per + performance
            "filename": str(file.absolute()),
            "config": None
        })
    return models

@app.get("/sdapi/v1/options")
async def get_options():
    """Legge le opzioni dal DB SQLite"""
    current_model = await get_setting("sd_model_checkpoint")
    return {
        "sd_model_checkpoint": current_model,
        "samples_save": True,
        "samples_format": "png"
    }

@app.post("/sdapi/v1/options")
async def set_options(req: Dict[str, Any]):
    """Salva le opzioni nel DB. OpenWebUI lo usa per cambiare modello."""
    for key, value in req.items():
        await set_setting(key, str(value))
        
        # Se cambia il modello, triggeriamo un reload asincrono preventivo?
        # Per ora no, lo facciamo lazy alla prossima generazione per semplicità.
        if key == "sd_model_checkpoint":
            logger.info(f"Opzione cambio modello ricevuta: {value}")
            
    return {"status": "ok"}

@app.post("/sdapi/v1/txt2img")
async def txt2img(req: Txt2ImgRequest):
    # 1. Determina il modello
    model_name = req.override_settings.get("sd_model_checkpoint")
    if not model_name:
        model_name = await get_setting("sd_model_checkpoint")
    
    # Controllo base se il file esiste (Robustezza)
    if not (settings.MODEL_DIR / model_name).exists():
        # Prova a vedere se esiste nella lista modelli, altrimenti errore o fallback
        logger.warning(f"Modello richiesto {model_name} non trovato, uso default")
    
    # 2. Salva job nel DB
    params_json = json.dumps(req.dict())
    async with aiosqlite.connect(settings.DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO jobs (params, model_name, status) VALUES (?, ?, ?) RETURNING id", 
            (params_json, model_name, "pending")
        )
        job_id = (await cursor.fetchone())[0]
        await db.commit()
    
    # 3. Esegui (Wait per compatibilità A1111)
    # In uno scenario reale useresti websocket o polling, ma A1111 è sincrono per txt2img
    await run_job_in_background(job_id, model_name, req.dict())
    
    # 4. Recupera risultato
    async with aiosqlite.connect(settings.DB_PATH) as db:
        async with db.execute("SELECT status, result, error_msg FROM jobs WHERE id = ?", (job_id,)) as cursor:
            status_res, result, error = await cursor.fetchone()
            
    if status_res == "failed":
        raise HTTPException(status_code=500, detail=f"Generation failed: {error}")
        
    return {
        "images": [result],
        "parameters": req.dict(),
        "info": json.dumps({"job_id": job_id, "model": model_name})
    }

