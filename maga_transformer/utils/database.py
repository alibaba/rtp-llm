from typing import Any, Dict, List, Union, Optional
from pathlib import PosixPath, Path
import json
import enum
import struct
import os
import logging
import re

import torch
from safetensors import safe_open

from maga_transformer.utils.time_util import Timer
import maga_transformer.utils.meta_pickler as meta_pickler


class FinetuneType(enum.Enum):
    pretrain = "pretrain"
    lora = "lora"
    ptuning = "ptuning"

class TrainType(enum.Enum):
    deepspeed = "deepspeed"
    megatron = "megatron"

class CkptType(enum.Enum):
    torch = "torch"
    safetensors = "safetensors"

class CkptFileInfo:

    """The abstract file for any type checkpoint file.

    """

    file_name: str
    metadata: Dict[str, Any]

    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int

    ckpt_type: CkptType
    finetune_type: FinetuneType
    train_type: TrainType
    

    def __init__(self, file_name: str, finetune_type: FinetuneType = FinetuneType.pretrain, 
                 train_type: TrainType = TrainType.deepspeed,
                 tp_size: int = 1, tp_rank: int = 1, pp_size: int = 1, pp_rank: int = 1) -> None:

        if file_name.endswith(('.safetensors')):
            self.ckpt_type = CkptType.safetensors
        elif file_name.endswith(('.pth', '.bin', '.pt')):
            self.ckpt_type = CkptType.torch
        else:
            raise Exception(f"unsupport file type : {file_name}")

        self.file_name = file_name
        self.finetune_type = finetune_type
        self.train_type = train_type
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.pp_size = pp_size
        self.pp_rank = pp_rank


    def set_metadata(self, metadata: Dict[str, Any]):
        self.metadata = metadata
    
    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata
    
    def get_tensor_names(self) -> List[str]:
        return [name for name in self.metadata.keys()]

    @property
    def pretrain_pp_tp(self):
        if self.finetune_type == FinetuneType.pretrain:
            return (self.pp_size, self.tp_size)
        return (1,1)
    
    def is_safetensor(self) -> bool:
        if self.ckpt_type == CkptType.safetensors:
            return True
        return False

    def __lt__(self, other):
        if not isinstance(other, CkptFileInfo):
            raise NotImplemented(f"other's type : {type(other)} is not CkptFileInfo")
        if self.pp_size < other.pp_size:
            return True
        if self.tp_size < other.tp_size:
            return True
        if self.finetune_type == FinetuneType.PRETRAIN and other.finetune_type != FinetuneType.PRETRAIN:
            return True
        if self.finetune_type != FinetuneType.PRETRAIN and other.finetune_type == FinetuneType.PRETRAIN:
            return False
        # 暂时不支持LoRA 和 PTuning 共存
        assert(self.finetune_type == other.finetune_type)
        return self.file_name < other.file_name


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

class BaseDatabase:
    def load_tensor(self, name: str, datatype: torch.dtype = torch.float16) -> List[torch.Tensor]:
        raise NotImplementedError

class ModuleDatabase(BaseDatabase):
    ref_model: Optional[torch.nn.Module] = None

    def __init__(self, ref_model: torch.nn.Module):
        self.ref_model = ref_model

    def load_tensor(self, name: str, datatype: torch.dtype = torch.float16) -> List[torch.Tensor]:
        weight_name: str = re.sub(r'\.\d+\.', lambda x: '[' + x.group(0)[1:-1] + '].', name)
        try:
            return [eval('self.ref_model.' + weight_name)]
        except AttributeError:
            raise Exception(f'No weight named {weight_name} in reference model')

class CkptDatabase(BaseDatabase):

    PretrainFileList : List[CkptFileInfo]
    FinetuneFileList : List[CkptFileInfo]
    LoraFileList: Dict[LoraConfig, List[CkptFileInfo]]

    finetune_type : FinetuneType
    tranin_type : TrainType

    def __init__(self, path: Optional[str]) -> None:

        if path is None:
            return
        
        self.PretrainFileList = []
        self.FinetuneFileList = []
        self.LoraFileList = {}

        if os.path.isfile(path):
            raise Exception(f"CkptDatabase needs directory contains checkpoint files")

        
        if self.is_megatron_ckpt(Path(path)):
            root_path, tp_size, pp_size = self._get_megatron_info(Path(path))
            for pp_rank in range(pp_size):
                for tp_rank in range(tp_size):
                    ckpt_file: Path = self._detect_ckpt_file(root_path, pp_rank, tp_rank, pp_size, tp_size)
                    ckpt = CkptFileInfo(file_name=str(ckpt_file.resolve()), tp_size=tp_size, tp_rank=tp_rank, 
                                                pp_size=pp_size, pp_rank=pp_rank, train_type=TrainType.megatron)
                    ckpt.set_metadata(self._load_meta(ckpt.file_name))
                    self.PretrainFileList.append(ckpt)

            self.finetune_type = FinetuneType.pretrain
            self.tranin_type = TrainType.megatron

        # standard HF safetensors
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
                        ckpt.set_metadata(self._load_meta(ckpt.file_name))
                        self.PretrainFileList.append(ckpt)
                break
            
    def _contains(self, path: Path):
        for info in self.PretrainFileList + self.FinetuneFileList:
            if Path(info.file_name).resolve() == path.resolve():
                return True
        return False

    def load_ptuning(self, ptuning_path: Optional[str]):
        if ptuning_path is None or not os.path.exists(ptuning_path):
            return
        for f in Path(ptuning_path).glob("pytorch_model.bin"):            
            if not self._contains(f):
                ckpt = CkptFileInfo(file_name=str(f), finetune_type=FinetuneType.ptuning)
                ckpt.set_metadata(self._load_meta(ckpt.file_name))
                self.FinetuneFileList.append(ckpt)

    def get_pretrain_tensor_names(self) -> List[str]:
        tensor_names = []
        for ckptfile in self.PretrainFileList:
            tensor_names.extend(ckptfile.get_tensor_names())
        
        for ckptfile in self.FinetuneFileList:
            tensor_names.extend(ckptfile.get_tensor_names())
            
        return tensor_names

    def get_lora_tensor_names(self, name: str) -> List[Any]:
        tensor_names = []
        for key, value in self.LoraFileList.items():
            if key.name == name:
                for ckptfile in value:
                    tensor_names.extend(ckptfile.get_tensor_names())
        return tensor_names

    def _load_meta(self, file: str) -> Dict[str, Any]:
        # https://huggingface.co/docs/safetensors/metadata_parsing
        if self.is_safetensor(file):
            meta = {}
            with safe_open(file, framework="pt") as f_:
                with open(file, 'rb') as f:
                    length_of_header = struct.unpack('<Q', f.read(8))[0]
                    header = f.read(length_of_header)
                    metadata = json.loads(header)
                for key in f_.keys():
                    meta[key] = metadata[key]['data_offsets'][0]
            return meta
        else:
            return torch.load(file, pickle_module=meta_pickler)

    def load_tensor(self, name: str, datatype: torch.dtype = torch.float16) -> List[torch.Tensor]:
        tensors = []
        for ckpt_file in self.PretrainFileList:
            if name in ckpt_file.get_tensor_names():
                tensors.append(self._load(name, ckpt_file, datatype))

        for ckpt_file in self.FinetuneFileList:
            if name in ckpt_file.get_tensor_names():
                tensors.append(self._load(name, ckpt_file, datatype))

        return tensors
    
    def load_lora_tensor(self, lora_name: str, tensor_name: str) -> List[torch.Tensor]:
        tensors = []
        for key, value in self.LoraFileList.items():
            if not key.name == lora_name:
                continue
            for ckpt_file in value:
                if tensor_name in ckpt_file.get_tensor_names():
                    tensors.append(self._load(tensor_name, ckpt_file))
        return tensors
    
    def _load(self, name: str, ckptfile: CkptFileInfo, datatype: str = torch.float16) -> torch.Tensor:

        path: str = ckptfile.file_name
        if ckptfile.is_safetensor():
            with safe_open(path, framework="pt") as f:
                return f.get_tensor(name).to(datatype)
        else:
            meta = ckptfile.metadata[name]
            def __preload_tensor_content(file, tensor, meta, storage_offset):
                tensor_offset = meta[1] * torch._utils._element_size(dtype)
                tensor_bytes = tensor.numel() * torch._utils._element_size(dtype)
                with Timer() as t:
                    with open(file, 'rb') as f:
                        f.seek(storage_offset + tensor_offset)
                        f.read(tensor_bytes)
            with open(path, 'rb') as f:
                size = os.path.getsize(path)
                if isinstance(path, PosixPath):
                    path = path.as_posix()
                overall_storage = torch.UntypedStorage.from_file(path, False, size)
                with torch.serialization._open_zipfile_reader(f) as zip_file_reader:
                    storage_args = meta[0]
                    dtype = storage_args[1].dtype
                    name = 'data/' + storage_args[2]
                    n_elements = storage_args[4]
                    n_bytes = n_elements * torch._utils._element_size(dtype)
                    storage_offset = zip_file_reader.get_record_offset(name)
                    storage = overall_storage[storage_offset:storage_offset + n_bytes]
                    typed_storage = torch.storage.TypedStorage(
                        wrap_storage=storage,
                        dtype=dtype,
                        _internal=True)
                    tensor = torch._utils._rebuild_tensor_v2(typed_storage, *meta[1:])
                    # preread tensor content to memory: avoid multi-thread read file (e.g. from Fuse) cause cache miss
                    __preload_tensor_content(path, tensor, meta, storage_offset)
                    tensor = tensor.contiguous().half()

                    return tensor

    def load_lora(self, name: str, lora_path: str):
        for key, _ in self.LoraFileList.items():
            if key.name == name:
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
        lora_config = LoraConfig(name, lora_config_path)
        ckpt = CkptFileInfo(file_name=str(lora_ckpt_paths[0]), finetune_type=FinetuneType.lora)
        ckpt.set_metadata(self._load_meta(ckpt.file_name))
        self.LoraFileList[lora_config] = [ckpt]
        self.lora_check()
    
    def lora_tensor_check(self, tensor_name: str):
        # check lora tensor names. the lora tensor name should match 
        # the format which is "base_model.model.{}.{}.weight"
        pattern = r'base_model\.model\.(.*)\.(lora_A|lora_B)\.weight'
        if re.fullmatch(pattern, tensor_name) == None:
            raise Exception(f"invalid lora tensor name : {tensor_name}")
        return None

    def lora_check(self) -> None:
        if len(self.LoraFileList) == 0:
            logging.info("The database has no lora ckpts")
            return
        
        for key, _ in self.LoraFileList.items():
            tensor_names = self.get_lora_tensor_names(key.name)
            for name in tensor_names:
                self.lora_tensor_check(name)
                

    def remove_lora(self, name:str):
        for key, _ in self.LoraFileList.items():
            if key.name == name:
                del self.LoraFileList[key]
                return True
        return False
    
    def get_lora_config(self, name: str):
        for key, _ in self.LoraFileList.items():
            if key.name == name:
                return key
        return LoraConfig()
    
    def has_lora(self):
        return len(self.LoraFileList) == 1
    
    def get_lora(self, name: str) -> List[Any]:
        for key, value in self.LoraFileList.items():
            if key.name == name:
                return value
        return []
    
    def get_first_lora_name(self):
        if len(self.LoraFileList) == 1:
            return list(self.LoraFileList)[0].name
        return None
    
    def save_ft(self):
        pass

    def is_empty(self):
        pass

    def is_safetensor(self, file: str) -> bool:
        if file.endswith(('.safetensors', ".safetensors")):
            return True
        else:
            return False

    def is_megatron_ckpt(self, ckpt_path: Path) -> bool:

        subdirs = list(ckpt_path.rglob(r"mp_rank_??"))
        if len(subdirs) > 0:
            return True
        subdirs = list(ckpt_path.rglob(r"mp_rank_??_???"))
        if len(subdirs) > 0:
            return True
        
        # Megatron 只输出global_stepxxx/mp_rank_xxx_model_states.pt,但是deepspeed还会输出*.bin
        patterns = ["*.pth", "*.bin", "*.pt"]
        ckpt_files = [file for pattern in patterns for file in ckpt_path.glob(pattern)]
        if len(ckpt_files) > 0:
            return False
        subdirs = list(ckpt_path.rglob(r"mp_rank_??_model_states.pt"))
        if len(subdirs) > 0:
            return True

        return False
    
    def _get_megatron_info(self, ckpt_path: Path):

        subdirs = list(ckpt_path.rglob(r"mp_rank_??"))
        if len(subdirs) > 0:
            tp = len(subdirs)
            pp = 1
            return subdirs[0].parents[0], len(subdirs), 1
        subdirs = list(ckpt_path.rglob(r"mp_rank_??_model_states.pt"))
        if len(subdirs) > 0:
            return subdirs[0].parents[0], len(subdirs), 1
        subdirs = list(ckpt_path.rglob(r"mp_rank_??_???"))
        assert len(subdirs) > 0, "magatron path must like: [mp_rank_?? | mp_rank_??_model_states.pt | mp_rank_??_???]"
        mp: int = len(subdirs)
        tp_ranks = set()
        pp_ranks = set()
        for subdir in map(str, subdirs):
            tp_rank = re.findall(r"mp_rank_(\d\d)_000$", subdir)
            pp_rank = re.findall(r"mp_rank_00_(\d\d\d)$", subdir)
            if len(tp_rank) > 0:
                tp_ranks.add(tp_rank[0])
            if len(pp_rank) > 0:
                pp_ranks.add(pp_rank[0])
        tp = len(tp_ranks)
        pp = len(pp_ranks)
        assert tp * pp == mp , f"tp:{tp} * pp{pp} != mp:{mp}"
        return subdirs[0].parents[0], tp, pp
    
    def _detect_ckpt_file(self, dir: str, pp_rank: int, tp_rank: int, pp_size: int, tp_size:int) -> Path:
        if pp_size == 1:
            suffix = f"mp_rank_{tp_rank:02}"
            if (dir / suffix).exists():
                path = dir / suffix
            else:
                assert((dir / f"mp_rank_{tp_rank:02}_model_states.pt").exists())
                path = dir
        else:
            suffix = f"mp_rank_{tp_rank:02}_{pp_rank:03}"
            path = dir / suffix

        if not path.exists():
            raise FileNotFoundError(f"{path.absolute()} not exists!")
        if (path / "model_rng.pt").exists():
            suffix = f"model_rng.pt"
        elif (path / "model_optim_rng.pt").exists():
            suffix = f"model_optim_rng.pt"
        elif (path /  f"mp_rank_{tp_rank:02}_model_states.pt").exists():
            suffix =  f"mp_rank_{tp_rank:02}_model_states.pt"
        path = path / suffix
        return path
    
    def get_megatron_ckpt_files(self, ckpt_path: Path) -> List[CkptFileInfo]:
        root_path, tp_size, pp_size = self._get_megatron_info(ckpt_path)
        ckpt_files: List[CkptFileInfo] = []
        for pp_rank in range(pp_size):
            for tp_rank in range(tp_size):
                ckpt_file: Path = self._detect_ckpt_file(root_path, pp_rank, tp_rank, pp_size, tp_size)
                ckpt_files.append(CkptFileInfo(
                    file_name=str(ckpt_file.resolve()), tp_size=tp_size, tp_rank=tp_rank, pp_size=pp_size, pp_rank=pp_rank, train_type=TrainType.megatron))
        return ckpt_files


    @property
    def pretrain_pp_tp(self):
        for pretrainfile in self.PretrainFileList:
            if pretrainfile.finetune_type == FinetuneType.pretrain:
                return (pretrainfile.pp_size, pretrainfile.tp_size)
        return (1,1)
