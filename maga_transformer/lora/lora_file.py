from typing import Any, Dict, List, Set, Union, Optional, NamedTuple
from pathlib import PosixPath, Path
import json
import os
import logging
import re
import torch

from maga_transformer.utils.ckpt_file_info import CkptFileInfo, FinetuneType
from maga_transformer.utils.dump_config_utils import dump_lora_infos_to_table

class LoraConfig:

    name: str
    path: str
    rank: int = 0
    lora_alpha: int = 0
    lora_dropout: float = 0.0
    target_modules: Union[List[str],str] = []
    is_merge: bool = False

    def __init__(self, name: str = '', json_path: str = '') -> None:

        if json_path.endswith('.json'):
            with open(json_path, 'r') as f:
                f = json.load(f)
                self.path = json_path
                self.rank = f['r']
                self.lora_alpha = f['lora_alpha']
                self.lora_dropout = f['lora_dropout']
                self.target_modules = f['target_modules']
                self.name = name

        return

    def get_scale(self):
        return self.lora_alpha / self.rank


class LoraInfos:

    class LoraInfo(NamedTuple):
        lora_name: str
        path: str
        target_modules: Union[List[str], str]
        tensor_nums: int


    lora_infos: List[LoraInfo]

    def __init__(self) -> None:
        self.lora_infos = []

    def add_lora_infos(self, LoraFileList : Dict[LoraConfig, List[CkptFileInfo]]):
        for key, value in LoraFileList.items():
            lora_name       = key.name
            path            = key.path
            target_modules  = key.target_modules
            tensor_nums     = sum([ckpt_file.tensor_num for ckpt_file in value])
            lora_info       = LoraInfos.LoraInfo(lora_name, path, target_modules, tensor_nums)
            self.lora_infos.append(lora_info)

    def dump(self):
        dump_lora_infos_to_table("DATABASE LORA INFOS", self.lora_infos)

class LoraCkpt:
    LoraFileList: Dict[LoraConfig, List[CkptFileInfo]]

    def __init__(self):
        self.LoraFileList = {}

    def get_lora_tensor_names(self, config_name: str) -> List[str]:
        tensor_names = []
        for key, value in self.LoraFileList.items():
            if key.name == config_name:
                for ckptfile in value:
                    tensor_names.extend(ckptfile.get_tensor_names())
        return tensor_names

    def load_lora_tensor(self, lora_name: str, tensor_name: str) -> List[torch.Tensor]:
        tensors = []
        for key, value in self.LoraFileList.items():
            if not key.name == lora_name:
                continue
            for ckpt_file in value:
                if tensor_name in ckpt_file.get_tensor_names():
                    tensors.append(ckpt_file.load_tensor(tensor_name))
        return tensors

    def load_lora(self, config_name: str, lora_path: str):
        for key, _ in self.LoraFileList.items():
            if key.name == config_name:
                raise Exception(f"load same name lora")
        # standard HF safetensors
        patterns = ["adapter_model.safetensors", "adapter_model.bin", "adapter_model.pth", "adapter_model.pt"]
        lora_ckpt_paths = []
        for pattern in patterns:
            lora_model_path = os.path.join(lora_path, pattern)
            if os.path.exists(lora_model_path):
                lora_ckpt_paths.append(lora_model_path)
        if len(lora_ckpt_paths) == 0:
            raise Exception(f"No lora model ckpt files")
        if len(lora_ckpt_paths) != 1:
            raise Exception(f"More than one model ckpt files")

        lora_config_path = os.path.join(lora_path, "adapter_config.json")
        if not os.path.exists(lora_config_path):
            raise Exception(f"Error lora config path: {lora_config_path}")

        logging.info(f"lora_config_path is {lora_config_path}")
        lora_config = LoraConfig(config_name, lora_config_path)
        ckpt = CkptFileInfo(file_name=str(lora_ckpt_paths[0]), finetune_type=FinetuneType.lora)
        self.LoraFileList[lora_config] = [ckpt]
        self.lora_check()

    def lora_check(self) -> None:
        if len(self.LoraFileList) == 0:
            logging.info("The database has no lora ckpts")
            return

        for key, _ in self.LoraFileList.items():
            tensor_names = self.get_lora_tensor_names(key.name)
            for name in tensor_names:
                self.lora_tensor_check(name)

    def lora_tensor_check(self, tensor_name: str):
        # check lora tensor names. the lora tensor name should match
        # the format which is "base_model.model.{}.{}.weight"
        pattern = r'base_model\.model\.(.*)\.(lora_A|lora_B)\.weight'
        if re.fullmatch(pattern, tensor_name) == None:
            raise Exception(f"invalid lora tensor name : {tensor_name}")

    def remove_lora(self, name:str):
        for key, _ in self.LoraFileList.items():
            if key.name == name:
                del self.LoraFileList[key]
                return True
        return False

    def has_lora(self):
        return len(self.LoraFileList) == 1

    def get_first_lora_name(self):
        if len(self.LoraFileList) == 1:
            return list(self.LoraFileList)[0].name
        return None

    def get_lora(self, config_name: str) -> List[Any]:
        for key, value in self.LoraFileList.items():
            if key.name == config_name:
                return value
        return []

    def get_lora_config(self, config_name: str):
        for key, _ in self.LoraFileList.items():
            if key.name == config_name:
                return key
        return LoraConfig()

    def dump_lora_info(self) -> None:
        lora_infos = LoraInfos()
        lora_infos.add_lora_infos(self.LoraFileList)
        lora_infos.dump()