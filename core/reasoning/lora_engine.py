# core/reasoning/lora_engine.py
from pathlib import Path

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    _ML_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment,misc]
    AutoModelForCausalLM = None  # type: ignore[assignment,misc]
    BitsAndBytesConfig = None  # type: ignore[assignment,misc]
    PeftModel = None  # type: ignore[assignment,misc]
    _ML_AVAILABLE = False

from core.config import settings


class LoRAEngine:
    """
    Manages a single base model instance with hot-swappable LoRA adapters.
    Adapters are loaded lazily on first use and kept in memory.
    """

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._loaded_adapters: set[str] = set()
        self._active_adapters: list[str] = []

    def _load_base_model(self):
        if self._model is not None:
            return
        if not _ML_AVAILABLE:
            raise RuntimeError("ML deps not installed (torch, transformers, peft)")
        print(f"Loading base model: {settings.BASE_MODEL_ID}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_ID)
        self._model = AutoModelForCausalLM.from_pretrained(
            settings.BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print("Base model loaded.")

    def load_adapter(self, adapter_name: str):
        """Load a LoRA adapter from disk into the model. No-op if already loaded."""
        if adapter_name in self._loaded_adapters:
            return
        self._load_base_model()
        adapter_path = Path(settings.ADAPTERS_DIR) / adapter_name
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        if not self._loaded_adapters:
            self._model = PeftModel.from_pretrained(
                self._model, str(adapter_path), adapter_name=adapter_name
            )
        else:
            self._model.load_adapter(str(adapter_path), adapter_name=adapter_name)

        self._loaded_adapters.add(adapter_name)
        print(f"Adapter loaded: {adapter_name}")

    def _set_active_adapters(self, adapter_names: list[str]):
        self._model.set_adapter(adapter_names)
        self._active_adapters = adapter_names

    def activate(self, adapter_names: list[str]):
        """Load (if needed) and activate the given adapters."""
        for name in adapter_names:
            self.load_adapter(name)
        self._set_active_adapters(adapter_names)

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        self._load_base_model()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded[len(self._tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]


# Singleton — one model instance per process
lora_engine = LoRAEngine()
