import enum
import json
import logging
import os
import struct
from pathlib import PosixPath
from typing import Any, Dict, List, Tuple

import torch
from safetensors import safe_open

import rtp_llm.utils.meta_pickler as meta_pickler
from rtp_llm.utils.time_util import Timer


class CkptType(enum.Enum):
    torch = "torch"
    safetensors = "safetensors"


class FinetuneType(enum.Enum):
    pretrain = "pretrain"
    lora = "lora"
    ptuning = "ptuning"


_SAFETENSORS_DTYPES: Dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I16": torch.int16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "F32": torch.float32,
    "I64": torch.int64,
    "F64": torch.float64,
}


class CkptFileInfo:
    """The abstract file for any type checkpoint file."""

    file_name: str
    metadata: Dict[str, Any]

    ckpt_type: CkptType
    finetune_type: FinetuneType

    def __init__(
        self, file_name: str, finetune_type: FinetuneType = FinetuneType.pretrain
    ) -> None:

        if file_name.endswith((".safetensors")):
            self.ckpt_type = CkptType.safetensors
        elif file_name.endswith((".pth", ".bin", ".pt")):
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

    @property
    def file_size(self) -> int:
        return os.path.getsize(self.file_name)

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
        if not hasattr(self, "_sorted_tensor_cache"):
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
        sorted_tensors = sorted(tensor_offsets, key=lambda x: x[1])

        return [name for name, _ in sorted_tensors]

    def _load_meta(self, file: str) -> Dict[str, Any]:
        # https://huggingface.co/docs/safetensors/metadata_parsing
        if self.is_safetensor():
            meta = {}
            with safe_open(file, framework="pt") as f_:
                with open(file, "rb") as f:
                    length_of_header = struct.unpack("<Q", f.read(8))[0]
                    header = f.read(length_of_header)
                    metadata = json.loads(header)
                for key in f_.keys():
                    meta[key] = metadata[key]["data_offsets"][0]
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

    def _get_safetensor_handle(self):
        if not hasattr(self, "_st_handle") or self._st_handle is None:
            self._st_handle = safe_open(self.file_name, framework="pt")
        return self._st_handle

    def close_safetensor_handle(self):
        if hasattr(self, "_st_handle") and self._st_handle is not None:
            self._st_handle.__exit__(None, None, None)
            self._st_handle = None

    def load_tensor(self, name: str, datatype: str = torch.float16) -> torch.Tensor:
        """Load a single tensor by name.

        Note: this method is NOT thread-safe — the cached safetensor handle
        is shared across calls and has no internal locking.
        """
        path: str = self.file_name
        if self.is_safetensor():
            f = self._get_safetensor_handle()
            return f.get_tensor(name).to(datatype)
        else:
            meta = self.metadata[name]

            def __preload_tensor_content(file, tensor, meta, storage_offset):
                tensor_offset = meta[1] * torch._utils._element_size(dtype)
                tensor_bytes = tensor.numel() * torch._utils._element_size(dtype)
                with Timer() as t:
                    with open(file, "rb") as f:
                        f.seek(storage_offset + tensor_offset)
                        f.read(tensor_bytes)

            with open(path, "rb") as f:
                size = os.path.getsize(path)
                if isinstance(path, PosixPath):
                    path = path.as_posix()
                overall_storage = torch.UntypedStorage.from_file(path, False, size)
                with torch.serialization._open_zipfile_reader(f) as zip_file_reader:
                    storage_args = meta[0]
                    dtype = storage_args[1].dtype
                    name = "data/" + storage_args[2]
                    n_elements = storage_args[4]
                    n_bytes = n_elements * torch._utils._element_size(dtype)
                    storage_offset = zip_file_reader.get_record_offset(name)
                    storage = overall_storage[storage_offset : storage_offset + n_bytes]
                    typed_storage = torch.storage.TypedStorage(
                        wrap_storage=storage, dtype=dtype, _internal=True
                    )
                    tensor = torch._utils._rebuild_tensor_v2(typed_storage, *meta[1:])
                    # preread tensor content to memory: avoid multi-thread read file (e.g. from Fuse) cause cache miss
                    __preload_tensor_content(path, tensor, meta, storage_offset)
                    tensor = tensor.contiguous().to(datatype)

                    return tensor

    def safetensors_header(self) -> Tuple[Dict[str, Any], int]:
        """Full per-tensor header, plus the offset where the data block starts.

        self.metadata only keeps each tensor's start offset; shape and dtype are needed
        to read a row range, so the header is parsed again here and cached.
        """
        if not self.is_safetensor():
            raise Exception(f"{self.file_name} is not a safetensors file")
        if getattr(self, "_st_header", None) is None:
            with open(self.file_name, "rb") as f:
                length_of_header = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(length_of_header))
            header.pop("__metadata__", None)
            self._st_header = (header, 8 + length_of_header)
        return self._st_header

    def load_tensors_by_row_shard(self, names: List[str], device: str, shard_fn):
        """Read the named tensors, keeping only this rank's dim-0 slice where asked.

        shard_fn(name) returns (num_shards, shard_index) for a tensor that has to be cut,
        or None to read it whole. safetensors stores tensors contiguously in C order, so a
        dim-0 row range is one contiguous byte range and can be read with a single pread
        -- the other shards' bytes never leave the disk.
        """
        header, data_start = self.safetensors_header()
        result: Dict[str, torch.Tensor] = {}
        fd = os.open(self.file_name, os.O_RDONLY)
        try:
            # Offset order keeps the access pattern sequential across the file.
            for name in sorted(names, key=lambda n: header[n]["data_offsets"][0]):
                meta = header[name]
                dtype_name = meta["dtype"]
                if dtype_name not in _SAFETENSORS_DTYPES:
                    raise Exception(
                        f"{self.file_name}: tensor {name} has unsupported dtype {dtype_name}"
                    )
                shape = list(meta["shape"])
                begin, end = meta["data_offsets"]
                offset, num_bytes = begin, end - begin

                shard = shard_fn(name) if shape else None
                if shard is not None:
                    num_shards, shard_index = shard
                    rows = shape[0]
                    if rows % num_shards != 0 or num_bytes % rows != 0:
                        raise Exception(
                            f"{self.file_name}: tensor {name} with shape {shape} and"
                            f" {num_bytes} bytes cannot be split into {num_shards} shards"
                        )
                    row_bytes = num_bytes // rows
                    shape[0] = rows // num_shards
                    num_bytes = shape[0] * row_bytes
                    offset = begin + shard_index * num_bytes

                buffer = torch.empty(num_bytes, dtype=torch.uint8)
                view = memoryview(buffer.numpy())
                # A single read(2) transfers at most 0x7ffff000 bytes, so tensors above
                # 2GiB come back short and have to be resumed.
                done = 0
                while done < num_bytes:
                    read = os.preadv(fd, [view[done:]], data_start + offset + done)
                    if read <= 0:
                        raise IOError(
                            f"{self.file_name}: tensor {name} short read, got {done}"
                            f" of {num_bytes} bytes at {data_start + offset}"
                        )
                    done += read
                result[name] = (
                    buffer.view(_SAFETENSORS_DTYPES[dtype_name])
                    .reshape(shape)
                    .to(device)
                )
        finally:
            os.close(fd)
        return result

    def load_tensors(self, device: str = "cuda", direct_io=True):
        file_path = os.path.abspath(self.file_name)
        if file_path.startswith(("/dev/shm", "/run/shm", "/sys/fs/cgroup")):
            logging.info(f"abs path : {file_path} cannot use direct_io")
            direct_io = False

        if self.is_safetensor():
            try:
                from fast_safetensors import load_safetensors_to_device

                use_shm = True
                logging.info(
                    f"use fast_safetensors to device: {device} direct_io:{direct_io} use_shm:{use_shm}"
                )
                res = load_safetensors_to_device(
                    self.file_name,
                    max_buf_size=2 * 1024 * 1024 * 1024,
                    direct_io=direct_io,
                    use_shm=use_shm,
                    device=device,
                )
                logging.debug("load_safetensors_to_device result: %s", list(res.keys()))
                return res
            except ModuleNotFoundError:
                logging.info(f"use safetensors to device: {device}")
                from safetensors.torch import load_file

                return load_file(self.file_name, device=device)
            except RuntimeError as e:
                logging.info(
                    f"use safetensors to device: {device} instead, because fast load failed: {e},"
                )
                from safetensors.torch import load_file

                return load_file(self.file_name, device=device)
        else:
            return torch.load(self.file_name, map_location=torch.device(device))

    def __lt__(self, other):
        if not isinstance(other, CkptFileInfo):
            raise NotImplemented(f"other's type : {type(other)} is not CkptFileInfo")
        if (
            self.finetune_type == FinetuneType.PRETRAIN
            and other.finetune_type != FinetuneType.PRETRAIN
        ):
            return True
        if (
            self.finetune_type != FinetuneType.PRETRAIN
            and other.finetune_type == FinetuneType.PRETRAIN
        ):
            return False
        # 暂时不支持LoRA 和 PTuning 共存
        assert self.finetune_type == other.finetune_type
        return self.file_name < other.file_name
