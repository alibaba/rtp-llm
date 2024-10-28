from typing import Dict, List
import torch
import os
import logging
from awq import AutoAWQForCausalLM

from maga_transformer.tools.quant.base_quanter import QUANT_TYPE, BaseQuanter
class AwqQuanter(BaseQuanter):
    def __init__(self, quantize_config: Dict[str, str], model_path: str, offload_folder: str):
        super().__init__()
        self.quantize_config = quantize_config
        max_memory = {}
        per_gpu_max_memory = int(torch.cuda.get_device_properties(torch.device('cuda:0')).total_memory*0.95/1024/1024/1024)
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        cuda_device_list = cuda_devices.split(',') if cuda_devices is not None else \
            [str(i) for i in range(torch.cuda.device_count())]
        max_memory.update({int(i): f'{per_gpu_max_memory}GIB' for i in range(len(cuda_device_list))})
        logging.info(f'max_memory: {max_memory}')

        model = AutoAWQForCausalLM.from_pretrained(model_path, True)
        self.model = model.eval().half()


    @classmethod
    def quant_type(cls):
        return QUANT_TYPE.AWQ

    def _quant(self, examples: List[Dict[str, torch.Tensor]]):
        examples = [_.get('input_ids').tolist() for _ in examples]
        self.model.quantize(
            tokenizer=None,
            quant_config=self.quantize_config,
            calib_data=examples
        )

    def _save_quantized(self, output_path: str):
        self.model.save_quantized(output_path)
