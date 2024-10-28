from enum import Enum
import logging
from typing import Any, Dict, List, Type
import torch


from maga_transformer.utils.time_util import Timer



class QUANT_TYPE(Enum):
    GPTQ = 'gptq'
    AWQ = 'awq'
    SMQ = 'smq'
    FP8 = 'fp8'  # fix later

    @classmethod
    def from_str(cls, value: str) -> 'QUANT_TYPE':
        lower_value = value.lower()
        for _, member in cls.__members__.items():
            if lower_value == member.value:
                return member
        raise ValueError('No enum member with value %s' % value)

    def to_str(self) -> str:
        return self.value



_quanter_factory: Dict[QUANT_TYPE, Type[Any]] = {}
def register_quanter(quant_type: QUANT_TYPE, quanter_type: Any):
    global _quanter_factory
    if quant_type in _quanter_factory and _quanter_factory[quant_type] != quanter_type:
        raise Exception(f"try register quanter failed, quant_type: {quant_type} quanter:{quanter_type} conflict with {_quanter_factory[quant_type]}")

    _quanter_factory[quant_type] = quanter_type


class BaseQuanter:
    @classmethod
    def register(cls):
        register_quanter(cls.quant_type(), cls)

    @classmethod
    def quant_type(cls):
        raise NotImplementedError("quant_type method is not implement")

    def quant(self, examples: List[Dict[str, torch.Tensor]]):
        with Timer() as t:
            self._quant(examples)
        logging.info(f'quantize model use:{t.cost_ms()/1000:.0f}s')

    def _quant(self, examples: List[Dict[str, torch.Tensor]]):
        raise NotImplementedError("quant method is not implement")

    def save_quantized_model(self, output_path: str):
        save_ret = False
        # output_path: str = fetch_remote_file_to_local(output_path, MountRwMode.RWMODE_RW)
        for _ in range(3):
            try:
                self._save_quantized(output_path)
                save_ret = True
                break
            except BaseException as e:
                logging.warn(f"save to {output_path} failed, e: {str(e)}")

        if not save_ret:
            raise Exception(f"save to {output_path} failed")

    def _save_quantized(self, output_path:str):
        raise NotImplementedError("save method is not implement")


class QuanterFactory:
    @staticmethod
    def get_quant_cls(quant_type:str):
        global _quanter_factory
        return _quanter_factory[quant_type]

    @staticmethod
    def create_quanter(quant_type: QUANT_TYPE, quantize_config: Dict[str, str], model_path: str, offload_folder: str):
        quanter_cls = QuanterFactory.get_quant_cls(quant_type)
        return quanter_cls(quantize_config, model_path, offload_folder)
