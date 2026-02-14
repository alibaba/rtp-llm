import logging
import threading
from typing import Any, Dict, Optional

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.async_decoder_engine.rpc_engine import LanguageCppEngine
from rtp_llm.lora.lora_exception import LoraCountException, LoraException
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.utils.time_util import Timer


class LoraManager:
    thread_lock_ = threading.Lock()
    lora_infos_: Dict[str, str]

    engine_: BaseEngine
    lora_cpp_wrapper_: Any
    weights_loader_: ModelLoader

    def __init__(
        self, engine: BaseEngine, max_lora_model_size: int = -1, local_rank: int = 0
    ) -> None:
        self.engine_ = engine
        self.lora_infos_ = {}
        self.max_lora_model_size_ = max_lora_model_size
        self.device: str = f"cuda:{local_rank}"
        assert isinstance(self.engine_, LanguageCppEngine)
        self.lora_cpp_wrapper_ = self.engine_.rtp_llm_op_.ft_op
        assert isinstance(self.engine_.model.model_weights_loader, ModelLoader)
        self.weights_loader_ = self.engine_.model.model_weights_loader
        with Timer() as timer:
            model_lora_infos = self.engine_.model.model_config.lora_infos
            if model_lora_infos is not None and len(model_lora_infos) > 1:
                logging.info(f"model_lora_infos is {model_lora_infos}")
                for key, value in model_lora_infos.items():
                    self.add_lora(key, value)
        logging.info(f"update lora weights time: {timer.cost_ms() / 1000 :.2f} s")

    def _check_loraInfo_size(self, lora_infos: Dict[str, str]):
        if (
            self.max_lora_model_size_ != -1
            and len(lora_infos) > self.max_lora_model_size_
        ):
            raise LoraCountException(
                f"lora_infos[{lora_infos}]'s size exceed MAX_LORA_MODEL_SIZE[{self.max_lora_model_size_}]"
            )

    def get_add_lora_map(self, lora_infos: Dict[str, str]) -> Dict[str, str]:
        with self.thread_lock_:
            self._check_loraInfo_size(lora_infos)
            add_lora_map: Dict[str, str] = {}
            for adapter_name, lora_path in lora_infos.items():
                if (
                    adapter_name not in self.lora_infos_
                    or lora_path != self.lora_infos_[adapter_name]
                ):
                    add_lora_map[adapter_name] = lora_path
            return add_lora_map

    def get_remove_lora_map(self, lora_infos: Dict[str, str]) -> Dict[str, str]:
        with self.thread_lock_:
            self._check_loraInfo_size(lora_infos)
            remove_lora_map: Dict[str, str] = {}
            for adapter_name, lora_path in self.lora_infos_.items():
                if (
                    adapter_name not in lora_infos
                    or lora_path != lora_infos[adapter_name]
                ):
                    remove_lora_map[adapter_name] = lora_path
            return remove_lora_map

    def add_lora(self, adapter_name: str, lora_path: str) -> Optional[LoraException]:
        with self.thread_lock_:
            assert adapter_name not in self.lora_infos_.keys()
            self.lora_infos_[adapter_name] = lora_path
            weights = self.weights_loader_.load_lora_weights(
                adapter_name, lora_path, "cpu"
            )
            self.lora_cpp_wrapper_.add_lora(
                adapter_name, weights.lora_a_weights, weights.lora_b_weights
            )

    def remove_lora(self, adapter_name: str) -> Optional[LoraException]:
        with self.thread_lock_:
            assert adapter_name in self.lora_infos_.keys()
            del self.lora_infos_[adapter_name]
            self.lora_cpp_wrapper_.remove_lora(adapter_name)
