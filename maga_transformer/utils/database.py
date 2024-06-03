from typing import Any, Dict, List, Set, Union, Optional, NamedTuple
from pathlib import PosixPath, Path
import json
import os
import logging
import re
import torch

from maga_transformer.utils.ckpt_file_info import CkptFileInfo, FinetuneType, TrainType
from maga_transformer.utils.megatron_util import MegatronUtil
from maga_transformer.utils.lora_ckpt import LoraCkpt, LoraConfig

class BaseDatabase:

    def get_pretrain_tensor_names(self) -> List[str]:
        raise NotImplementedError

    def get_lora_tensor_names(self, name: str) -> List[str]:
        raise NotImplementedError
    
    def load_tensor(self, name: str, datatype: torch.dtype = torch.float16) -> List[torch.Tensor]:
        raise NotImplementedError

class ModuleDatabase(BaseDatabase):
    ref_module: Optional[torch.nn.Module] = None

    def __init__(self, ref_module: torch.nn.Module):
        self.ref_module = ref_module

    def load_tensor(self, name: str, datatype: torch.dtype = torch.float16) -> List[torch.Tensor]:
        #TODO(xinfei.sxf) add comment for this regex
        weight_name: str = re.sub(r'\.\d+\.', lambda x: '[' + x.group(0)[1:-1] + '].', name)
        try:
            return [eval('self.ref_module.' + weight_name).to(dtype = datatype)]
        except AttributeError:
            raise Exception(f'No weight named {weight_name} in reference module')
    
    def get_pretrain_tensor_names(self) -> List[str]:
        return list(self.ref_module.state_dict().keys())
        
class DictDatabase(BaseDatabase):
    ref_dict: Dict[str, torch.Tensor] = {}

    def __init__(self, ref_dict: Dict[str, torch.Tensor]):
        self.ref_dict = ref_dict
    
    def load_tensor(self, name: str, datatype: torch.dtype = torch.float16) -> List[torch.Tensor]:
        try:
            return [self.ref_dict[name].to(dtype = datatype)]
        except KeyError:
            raise Exception(f'No weight named {name} in dict')

    def get_pretrain_tensor_names(self) -> List[str]:
        return list(self.ref_dict.keys())    

class CkptDatabase(BaseDatabase):

    PretrainFileList : List[CkptFileInfo]
    FinetuneFileList : List[CkptFileInfo]
    LoraCkpt: LoraCkpt

    finetune_type : FinetuneType
    tranin_type : TrainType

    def __init__(self, path: Optional[str], ptuning_path: Optional[str] = None) -> None:

        if path is None:
            return
        
        self.PretrainFileList = []
        self.FinetuneFileList = []
        self.LoraCkpt = LoraCkpt()
        
        if os.path.isfile(path):
            raise Exception(f"CkptDatabase needs directory contains checkpoint files")

        if MegatronUtil.is_megatron_ckpt(Path(path)):
            self.load_megatron_meta(path)
        else:
            self.load_hf_meta(path)

        self.load_ptuning_meta(ptuning_path)

    def load_megatron_meta(self, path: str):
        self.PretrainFileList = self.get_megatron_ckpt_files(Path(path))
        self.finetune_type = FinetuneType.pretrain
        self.tranin_type = TrainType.megatron

    def get_megatron_ckpt_files(self, ckpt_path: Path) -> List[CkptFileInfo]:
        root_path, tp_size, pp_size = MegatronUtil.get_megatron_info(ckpt_path)
        ckpt_files: List[CkptFileInfo] = []
        for pp_rank in range(pp_size):
            for tp_rank in range(tp_size):
                ckpt_file: Path = MegatronUtil.detect_ckpt_file(root_path, pp_rank, tp_rank, pp_size, tp_size)
                ckpt_files.append(CkptFileInfo(
                    file_name=str(ckpt_file.resolve()), tp_size=tp_size, tp_rank=tp_rank, pp_size=pp_size, pp_rank=pp_rank, train_type=TrainType.megatron))
        return ckpt_files

    def load_hf_meta(self, path: str):
        # standard HF
        patterns = ["*.safetensors", "*.bin", "*.pth", "*.pt"]
        glob_files = {}

        for pattern in patterns:
            glob_files[pattern] = [file for file in Path(path).glob(pattern)]

        for _, value in glob_files.items():
            if len(value) != 0:
                exclude_pattern: re.Pattern[str] = re.compile(r'.*adapter_model\.bin.*|.*training_args\.bin.*')
                for f in value:
                    if not exclude_pattern.match(f.name):
                        ckpt = CkptFileInfo(file_name=str(f))
                        self.PretrainFileList.append(ckpt)
                break
            
    def load_ptuning_meta(self, ptuning_path: Optional[str]):
        if ptuning_path is None or not os.path.exists(ptuning_path):
            return
        for f in Path(ptuning_path).glob("pytorch_model.bin"):            
            if not self._contains(f):
                ckpt = CkptFileInfo(file_name=str(f), finetune_type=FinetuneType.ptuning)
                self.FinetuneFileList.append(ckpt)
            
    def _contains(self, path: Path):
        for info in self.PretrainFileList + self.FinetuneFileList:
            if Path(info.file_name).resolve() == path.resolve():
                return True
        return False

    def get_pretrain_tensor_names(self) -> List[str]:
        tensor_names = []
        for ckptfile in self.PretrainFileList:
            tensor_names.extend(ckptfile.get_tensor_names())
        
        for ckptfile in self.FinetuneFileList:
            tensor_names.extend(ckptfile.get_tensor_names())
            
        return tensor_names

    def load_tensor(self, name: str, datatype: torch.dtype = torch.float16) -> List[torch.Tensor]:
        tensors = []
        for ckpt_file in self.PretrainFileList:
            if name in ckpt_file.get_tensor_names():
                tensors.append(ckpt_file.load_tensor(name, datatype))

        for ckpt_file in self.FinetuneFileList:
            if name in ckpt_file.get_tensor_names():
                tensors.append(ckpt_file.load_tensor(name, datatype))

        return tensors

    @property
    def pretrain_pp_tp(self):
        for pretrainfile in self.PretrainFileList:
            if pretrainfile.finetune_type == FinetuneType.pretrain:
                return (pretrainfile.pp_size, pretrainfile.tp_size)
        return (1,1)
    
    def get_lora_tensor_names(self, config_name: str) -> List[str]:
        return self.LoraCkpt.get_lora_tensor_names(config_name)
    
    def load_lora_tensor(self, lora_name: str, tensor_name: str) -> List[torch.Tensor]:
        return self.LoraCkpt.load_lora_tensor(lora_name, tensor_name)

    def load_lora(self, config_name: str, lora_path: str):
        self.LoraCkpt.load_lora(config_name, lora_path)
                
    def remove_lora(self, name: str):
        return self.LoraCkpt.remove_lora(name)
    
    def get_lora_config(self, config_name: str):
        return self.LoraCkpt.get_lora_config(config_name)
    
    def has_lora(self):
        return self.LoraCkpt.has_lora()
    
    def get_first_lora_name(self):
        return self.LoraCkpt.get_first_lora_name()
    
    def dump_lora_info(self) -> None:
        self.LoraCkpt.dump_lora_info()