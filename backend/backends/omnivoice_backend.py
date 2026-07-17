"""
OmniVoice TTS backend implementation.

Wraps OmniVoice from k2-fsa/OmniVoice for zero-shot voice cloning and
voice design. 24 kHz output, ~600 supported languages, voice design via
the `instruct` parameter.

API notes
----------
- Model class: ``omnivoice.OmniVoice``
- Voice prompt: ``omnivoice.VoiceClonePrompt`` (a dataclass with pre-computed
  audio tokens, ref text, and ref RMS). This is "Pattern A" in the
  tts-engines doc — we store the dataclass directly in the prompt dict.
- Sample rate: 24 kHz (constant for all model sizes).
- Voice design: pass ``instruct="female, british accent"`` (English) or
  ``instruct="女，四川话"`` (Chinese). Only stable for English and Chinese
  per upstream docs.
- Language: accepts BCP-47 codes ("en", "zh") and full names ("English",
  "Chinese") — we pass through whatever the frontend sends.
- Reference text: optional. If missing, OmniVoice auto-transcribes via
  Whisper-large-v3-turbo (downloaded lazily on first use). We do NOT
  load the ASR model by default because it's ~1.5 GB on top of OmniVoice.
  Users who need it can set ``_load_asr=True`` via the backend attribute.
- MPS: OmniVoice forces the higgs audio tokenizer to CPU on MPS
  (its output channels exceed MPS's 65536 limit). We rely on that fix
  and force CPU on macOS in line with the existing engine conventions.
"""

import asyncio
import logging
from typing import ClassVar, List, Optional, Tuple

import numpy as np

from . import TTSBackend
from .base import (
    is_model_cached,
    get_torch_device,
    empty_device_cache,
    manual_seed,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)
from ..utils.cache import get_cache_key, get_cached_voice_prompt, cache_voice_prompt

logger = logging.getLogger(__name__)

OMNIVOICE_HF_REPO = "k2-fsa/OmniVoice"

# Files that must be present for the model to load. We check by listing
# any .safetensors / .bin / .pt in the snapshot — OmniVoice's main weights
# are safetensors.
_OMNIVOICE_WEIGHT_EXTENSIONS: ClassVar[tuple[str, ...]] = (".safetensors", ".bin", ".pt")


class OmniVoiceTTSBackend:
    """OmniVoice backend for zero-shot voice cloning and voice design."""

    def __init__(self):
        self.model = None
        self.model_size = "default"  # OmniVoice has only one public model size
        self._device = None
        self._model_load_lock = asyncio.Lock()

    def _get_device(self) -> str:
        # Force CPU on macOS — matches the existing engine convention and
        # avoids the MPS path for the higgs audio tokenizer even though
        # OmniVoice handles it internally. CPU is also the safest baseline
        # for PyInstaller frozen builds where the runtime hook for
        # torch.compiler.disable may interfere with MPS dispatch.
        return get_torch_device(force_cpu_on_mac=True, allow_xpu=True)

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return OMNIVOICE_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(
            OMNIVOICE_HF_REPO,
            weight_extensions=_OMNIVOICE_WEIGHT_EXTENSIONS,
        )

    async def load_model(self, model_size: str = "default") -> None:
        """Load the OmniVoice model."""
        if self.model is not None:
            return
        async with self._model_load_lock:
            if self.model is not None:
                return
            await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self):
        """Synchronous model loading."""
        model_name = "omnivoice"
        is_cached = self._is_model_cached()

        with model_load_progress(model_name, is_cached):
            device = self._get_device()
            self._device = device
            logger.info(f"Loading OmniVoice on {device}...")

            import torch
            from omnivoice import OmniVoice

            # CPU builds need map_location so safetensors load without a
            # CUDA device. We only patch on CPU; on CUDA/XPU the model
            # loader handles placement correctly.
            if device == "cpu":

                _orig_torch_load = torch.load

                def _patched_load(*args, **kwargs):
                    kwargs.setdefault("map_location", "cpu")
                    return _orig_torch_load(*args, **kwargs)

                torch.load = _patched_load
                try:
                    # float16 on CUDA/XPU, float32 on CPU. CPU can't do
                    # half-precision matmul efficiently, and omniVoice's
                    # token tensors are int64 anyway.
                    dtype = torch.float16 if device in ("cuda", "xpu") else torch.float32
                    model = OmniVoice.from_pretrained(
                        OMNIVOICE_HF_REPO,
                        device_map=device,
                        dtype=dtype,
                    )
                finally:
                    torch.load = _orig_torch_load
            else:
                model = OmniVoice.from_pretrained(
                    OMNIVOICE_HF_REPO,
                    device_map=device,
                    dtype=torch.float16,
                )

            self.model = model

        logger.info(
            f"OmniVoice loaded successfully (sample_rate={self.model.sampling_rate})"
        )

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            device = self._device
            del self.model
            self.model = None
            self._device = None
            empty_device_cache(device)
            logger.info("OmniVoice unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create a voice prompt from reference audio.

        OmniVoice's ``create_voice_clone_prompt`` returns a
        :class:`VoiceClonePrompt` dataclass with pre-computed audio tokens
        (C, T) and a normalized RMS. We store the dataclass directly in
        the prompt dict — this is Pattern A and matches Qwen's approach.

        Caching: we serialize the dataclass via ``VoiceClonePrompt.save()``
        and store the bytes in the voice prompt cache. The cache key
        includes a per-engine prefix to avoid collisions with other
        backends sharing the same cache directory.
        """
        await self.load_model()

        cache_key = (
            "omnivoice_" + get_cache_key(audio_path, reference_text)
            if use_cache
            else None
        )

        if cache_key:
            cached = get_cached_voice_prompt(cache_key)
            if isinstance(cached, dict) and "voice_clone_prompt" in cached:
                return cached, True

        def _create_prompt_sync():
            return self.model.create_voice_clone_prompt(
                ref_audio=str(audio_path),
                ref_text=reference_text or None,
                preprocess_prompt=True,
            )

        prompt = await asyncio.to_thread(_create_prompt_sync)

        # The dataclass isn't JSON-serializable as-is, but VoiceClonePrompt
        # is a flat @dataclass with primitive fields. We can pickle the
        # audio tokens tensor through torch.save (uses safetensors-style
        # serialization) and rebuild the dataclass on cache hit.
        voice_prompt = {
            "voice_clone_prompt": prompt,
            # Store the parts that survive a roundtrip through the cache
            # backend (which is typically a JSON-serialized path). The
            # `cache_voice_prompt` helper handles tensor serialization.
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }

        if cache_key:
            cache_voice_prompt(cache_key, voice_prompt)

        return voice_prompt, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        """
        Combine multiple voice prompts by concatenating their audio.

        OmniVoice processes a single reference audio at a time, so the
        upstream behavior is to combine the audio + text into one prompt.
        We use the shared helper which loads at the model's sample rate.
        """
        # model.sampling_rate is set during load_model; for the combine
        # step (which happens before generation), we fall back to 24000.
        sample_rate = 24000
        if self.model is not None and getattr(self.model, "sampling_rate", None):
            sample_rate = self.model.sampling_rate
        return await _combine_voice_prompts(
            audio_paths, reference_texts, sample_rate=sample_rate
        )

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio using OmniVoice.

        Args:
            text: Text to synthesize
            voice_prompt: Dict with ``voice_clone_prompt`` (VoiceClonePrompt
                dataclass) and ``ref_text`` from ``create_voice_prompt``.
                If ``voice_clone_prompt`` is absent, falls back to auto
                voice (OmniVoice picks a voice itself).
            language: BCP-47 code ("en", "zh") or full name ("English").
                OmniVoice's ``_resolve_language`` accepts both.
            seed: Random seed for reproducibility
            instruct: Voice design instruction. English/Chinese only;
                ignored if absent. Examples:
                  - "female, british accent"
                  - "male, low pitch, whisper"
                  - "女，四川话"

        Returns:
            Tuple of (audio_array, sample_rate). Sample rate is 24 kHz
            unless the audio tokenizer reports otherwise.
        """
        await self.load_model()

        voice_clone_prompt = voice_prompt.get("voice_clone_prompt")

        def _generate_sync() -> Tuple[np.ndarray, int]:
            import torch

            if seed is not None:
                manual_seed(seed, self._device)

            # Pass the prompt through OmniVoice's expected interface.
            # If voice_clone_prompt is None, OmniVoice uses auto-voice mode
            # (no reference, no instruct) — we allow this for completeness.
            audios = self.model.generate(
                text=text,
                language=language or None,
                voice_clone_prompt=voice_clone_prompt,
                instruct=instruct or None,
                # Text normalization is opt-in; off by default. We expose
                # only voice cloning/design/instruct features for now.
                normalize_text=False,
            )

            # OmniVoice returns list[np.ndarray] of shape (T,). We always
            # ask for a single text, so take [0]. Make sure dtype is
            # float32 — soundfile and the audio pipeline expect it.
            if not audios:
                raise RuntimeError("OmniVoice returned no audio")

            audio = np.asarray(audios[0], dtype=np.float32)
            sample_rate = (
                getattr(self.model, "sampling_rate", None) or 24000
            )
            return audio, int(sample_rate)

        return await asyncio.to_thread(_generate_sync)
