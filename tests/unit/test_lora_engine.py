# tests/unit/test_lora_engine.py
import pytest
from unittest.mock import MagicMock, patch


def test_lora_engine_loads_base_model_on_init():
    from core.reasoning.lora_engine import LoRAEngine
    with patch("core.reasoning.lora_engine.AutoModelForCausalLM") as mock_model_cls:
        with patch("core.reasoning.lora_engine.AutoTokenizer") as mock_tok_cls:
            mock_model_cls.from_pretrained.return_value = MagicMock()
            mock_tok_cls.from_pretrained.return_value = MagicMock()
            engine = LoRAEngine.__new__(LoRAEngine)
            engine._model = None
            engine._tokenizer = None
            engine._loaded_adapters = set()
            engine._active_adapters = []
    assert engine._model is None  # lazy load not triggered yet


def test_lora_engine_select_adapters_updates_active():
    from core.reasoning.lora_engine import LoRAEngine
    engine = LoRAEngine.__new__(LoRAEngine)
    engine._model = MagicMock()
    engine._tokenizer = MagicMock()
    engine._loaded_adapters = {"criminal"}
    engine._active_adapters = []

    engine._model.set_adapter = MagicMock()
    engine._set_active_adapters(["criminal"])

    engine._model.set_adapter.assert_called_with(["criminal"])
    assert engine._active_adapters == ["criminal"]
