import requests
import os
from pydantic import BaseModel, Field
from typing import Callable, Any


class Tools:
    class Valves(BaseModel):
        # Configurazione modificabile dalla UI
        CHATTERBOX_BACKEND_URL: str = Field(
            default="http://fast-chatterbox:8000/store-voice",
            description="L'endpoint del tuo backend custom che accetta il .wav",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def clone_voice_from_attachment(
        self,
        __files__,
        __event_emitter__,
    ) -> str:
        """
        Clona una voce basandosi su un file audio (WAV) caricato e restituisce il nome dello speaker creato.
        Da usare quando l'utente chiede di clonare una voce da un file allegato.

        :param file_id: L'ID del file caricato nel messaggio (fornito dall'LLM).
        """
        
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Salvataggio file audio...",
                    "done": False,
                },
            }
        )

        try:
            data = {"wav_path": __files__[0]["file"]["path"]}
            print(f"## Invio .wav al backend: {self.valves.CHATTERBOX_BACKEND_URL}")
            backend_response = requests.post(
                self.valves.CHATTERBOX_BACKEND_URL, json=data, timeout=120
            )
            if backend_response.status_code == 200:
                result = backend_response.json()
                speaker = {result.get("speaker", "Sconosciuto")}
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Campione salvato: {speaker}",
                            "done": True,
                        },
                    }
                )
                return f"Successo! Ho creato lo speaker. Il nome da impostare nei settings è: *{speaker}*"
            else:
                error_msg = f"Errore dal backend custom: {backend_response.status_code} - {backend_response.text}"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Errore durante la clonazione",
                            "done": True,
                        },
                    }
                )
                return error_msg

        except Exception as e:
            return f"Si è verificato un errore critico durante il processo: {str(e)}"

