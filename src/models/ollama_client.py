"""
Ollama Client
─────────────
Thin wrapper around the Ollama REST API.
Handles: text generation, vision queries, embeddings, streaming.
"""

import json
import logging
import requests
from typing import Generator, Optional

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Unified client for all Ollama model interactions.
    """

    def __init__(self, config):
        self.cfg = config.ollama
        self.base_url = self.cfg.base_url.rstrip("/")
        self._check_connection()

    # Connection

    def _check_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            logger.info(f"Ollama connected. Available models: {models}")
        except Exception as e:
            logger.warning(f"Ollama connection check failed: {e}. Make sure Ollama is running.")

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def is_model_available(self, model_name: str) -> bool:
        available = self.list_models()
        return any(model_name in m for m in available)

    # Text Generation

    def query(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Single-turn text completion."""
        model = model or self.cfg.llm_model
        payload = self._build_payload(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature or self.cfg.temperature,
            max_tokens=max_tokens or self.cfg.max_tokens,
            stream=False,
        )

        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.cfg.timeout,
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama timeout after {self.cfg.timeout}s")
        except Exception as e:
            raise RuntimeError(f"Ollama query failed: {e}")

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """Streaming text generation — yields tokens."""
        model = model or self.cfg.llm_model
        payload = self._build_payload(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature or self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            stream=True,
        )

        with requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=self.cfg.timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break

    # Vision (LLaVA)

    def vision_query(
        self,
        prompt: str,
        image_base64: str,
        model: Optional[str] = None,
    ) -> str:
        """
        Send image + text prompt to LLaVA vision model.
        image_base64: base64-encoded image string
        """
        model = model or self.cfg.vision_model
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 512,
            },
        }

        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.cfg.timeout * 2,  # Vision takes longer
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"Vision query failed: {e}")

    # Embeddings

    def embed(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        Uses nomic-embed-text by default.
        """
        model = model or self.cfg.embed_model
        embeddings = []

        for text in texts:
            try:
                r = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=30,
                )
                r.raise_for_status()
                emb = r.json().get("embedding", [])
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Embedding failed for text: {e}")
                embeddings.append([0.0] * 768)  # Fallback zero vector

        return embeddings

    # Helpers

    @staticmethod
    def _build_payload(
        model: str,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 4096,          # Context window
                "num_gpu": -1,            # Use all available GPU layers
                "num_thread": 4,          # CPU threads for remaining layers
            },
        }
        if system:
            payload["system"] = system
        return payload

    def pull_model(self, model_name: str, progress_callback=None) -> bool:
        """Pull a model from Ollama registry."""
        try:
            with requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line and progress_callback:
                        data = json.loads(line)
                        status = data.get("status", "")
                        progress_callback(status)
                        if status == "success":
                            return True
            return True
        except Exception as e:
            logger.error(f"Model pull failed: {e}")
            return False
