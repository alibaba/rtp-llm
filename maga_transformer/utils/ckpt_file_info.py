from typing import Any, Dict, List
from pathlib import PosixPath
import json
import enum
import os
import logging
import torch
import struct
from safetensors import safe_open

from maga_transformer.utils.time_util import Timer
import maga_transformer.utils.meta_pickler as meta_pickler

class CkptType(enum.Enum):
    torch = "torch"
    safetensors = "safetensors"

class FinetuneType(enum.Enum):
    pretrain = "pretrain"
    lora = "lora"
    ptuning = "ptuning"


class CkptFileInfo:

    """The abstract file for any type checkpoint file.

    """

    file_name: str
    metadata: Dict[str, Any]

    ckpt_type: CkptType
    finetune_type: FinetuneType
    

    def __init__(self, file_name: str, finetune_type: FinetuneType = FinetuneType.pretrain) -> None:

        if file_name.endswith(('.safetensors')):
            self.ckpt_type = CkptType.safetensors
        elif file_name.endswith(('.pth', '.bin', '.pt')):
            self.ckpt_type = CkptType.torch
        else:
            raise Exception(f"unsupport file type : {file_name}")

        self.file_name = file_name
        self.finetune_type = finetune_type
        self._load_meta(self.file_name)
    
    def get_tensor_names(self) -> List[str]:
        return [name for name in self.metadata.keys()]

    @property
    def tensor_num(self) -> int:
        return len(self.metadata.keys())
    
    def is_safetensor(self) -> bool:
        if self.ckpt_type == CkptType.safetensors:
            return True
        return False

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

    def get_tensor_read_order(self, name: str) -> List[str]:
        """
        获取推荐的张量读取顺序，基于物理存储位置优化I/O效率
        
        返回:
            List[str]: 按实际存储位置排序的张量名称列表
            
        抛出:
            RuntimeError: 如果文件元数据未正确加载
        """
        if not hasattr(self, '_sorted_tensor_cache'):
            # 延迟初始化排序缓存
            self._sorted_tensor_cache = self._build_sorted_tensor_list()
        
        return self._sorted_tensor_cache.index(name)
    
    def _build_sorted_tensor_list(self) -> List[str]:
        """构建按物理存储位置排序的张量列表"""
        if not self.metadata:
            raise RuntimeError("元数据未加载，请先调用 _load_meta")
            
        if self.is_safetensor():
            # 对safetensors按文件偏移量排序
            return self._safetensors_read_order()
        else:
            # 对其他格式使用文件中的自然顺序
            return self.get_tensor_names()
    
    def _safetensors_read_order(self) -> List[str]:
        """处理safetensors的物理存储顺序"""
        # 提取带偏移量的元组列表 (tensor_name, offset)
        tensor_offsets = [
            (name, self.metadata[name]) 
            for name in self.metadata
            if isinstance(self.metadata[name], int)
        ]
        
        # 按偏移量升序排列
        sorted_tensors = sorted(
            tensor_offsets, 
            key=lambda x: x[1]
        )
        
        return [name for name, _ in sorted_tensors]

    def _load_meta(self, file: str) -> Dict[str, Any]:
        # https://huggingface.co/docs/safetensors/metadata_parsing
        if self.is_safetensor():
            meta = {}
            with safe_open(file, framework="pt") as f_:
                with open(file, 'rb') as f:
                    length_of_header = struct.unpack('<Q', f.read(8))[0]
                    header = f.read(length_of_header)
                    metadata = json.loads(header)
                for key in f_.keys():
                    meta[key] = metadata[key]['data_offsets'][0]
            self.metadata = meta
        else:
            self.metadata = torch.load(file, pickle_module=meta_pickler)
    
    def get_tensor_type(self, tensor_name: str) -> torch.dtype:
        file: str = self.file_name
        if self.is_safetensor():
            with safe_open(file, framework="pt") as f:
                if tensor_name not in f.keys():
                    raise KeyError(f"Tensor '{tensor_name}' not found in the file")
                tensor = f.get_tensor(tensor_name)
                return tensor.dtype
        else:
            data = torch.load(file, map_location="meta")
            if tensor_name not in data:
                raise KeyError(f"Tensor '{tensor_name}' not found in the file")
            return data[tensor_name].dtype

    def load_tensor(self, name: str, datatype: str = torch.float16) -> torch.Tensor:
        path: str = self.file_name
        if self.is_safetensor():
            with safe_open(path, framework="pt") as f:
                return f.get_tensor(name).to(datatype)
        else:
            meta = self.metadata[name]
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
                    tensor = tensor.contiguous().to(datatype)

                    return tensor


    def load_tensors(self, device: str = "cuda:0", direct_io=True):
        file_path = os.path.abspath(self.file_name)
        if file_path.startswith(('/dev/shm', '/run/shm', '/sys/fs/cgroup')):
            logging.info(f"abs path : {file_path} cannot use direct_io")
            direct_io=False
                
        if self.is_safetensor():
            try:
                from fast_safetensors import load_safetensors_to_device
                use_shm = True
                logging.info(f"use fast_safetensors to device: {device} direct_io:{direct_io} use_shm:{use_shm}")
                res =  load_safetensors_to_device(self.file_name, max_buf_size=2*1024*1024*1024, direct_io=direct_io, use_shm=use_shm, device=device)
                logging.debug(f"load_safetensors_to_device result: {list(res.keys())}")
                return res
            except ModuleNotFoundError:
                logging.info(f"use safetensors to device: {device}")
                from safetensors.torch import load_file
                return load_file(self.file_name, device=device)
        else:
            return torch.load(self.file_name, map_location=torch.device(device))


    def __lt__(self, other):
        if not isinstance(other, CkptFileInfo):
            raise NotImplemented(f"other's type : {type(other)} is not CkptFileInfo")
        if self.finetune_type == FinetuneType.PRETRAIN and other.finetune_type != FinetuneType.PRETRAIN:
            return True
        if self.finetune_type != FinetuneType.PRETRAIN and other.finetune_type == FinetuneType.PRETRAIN:
            return False
        # 暂时不支持LoRA 和 PTuning 共存
        assert(self.finetune_type == other.finetune_type)
        return self.file_name < other.file_name
