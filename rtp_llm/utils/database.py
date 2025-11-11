import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import torch
from tqdm.auto import tqdm

from rtp_llm.lora.lora_file import LoraCkpt
from rtp_llm.utils.ckpt_file_info import CkptFileInfo, FinetuneType


class BaseDatabase:

    def get_pretrain_tensor_names(self) -> List[str]:
        raise NotImplementedError

    def get_lora_tensor_names(self, name: str) -> List[str]:
        raise NotImplementedError

    def load_tensor(
        self, name: str, data_type: Optional[torch.dtype] = torch.float16
    ) -> List[torch.Tensor]:
        raise NotImplementedError

    def get_tensor_order(self, name: str) -> List[int]:
        raise NotImplementedError

    def get_tensor_type(self, name: str) -> torch.dtype:
        raise NotImplementedError
    
    def get_max_file_size(self) -> int:
        raise NotImplementedError
    
    @property
    def is_safetensor(self) -> bool:
        return False

    @property
    def is_ft_style(self) -> bool:
        return False

    @property
    def ft_weight_params(self) -> Optional[Dict[str, Any]]:
        return None


class CkptDatabase(BaseDatabase):

    pretrain_file_list: List[CkptFileInfo]
    finetune_file_list: List[CkptFileInfo]
    lora_ckpt: LoraCkpt

    finetune_type: FinetuneType

    def __init__(self, path: Optional[str], ptuning_path: Optional[str] = None) -> None:

        if path is None:
            return

        self.pretrain_file_list = []
        self.finetune_file_list = []
        self.lora_ckpt = LoraCkpt()

        if os.path.isfile(path):
            raise Exception(f"CkptDatabase needs directory contains checkpoint files")

        self.load_hf_meta(path)

        self.load_ptuning_meta(ptuning_path)

        self._is_ft_style: bool = self._parse_weight_style(path)

        self._ft_weight_params = (
            self._parse_ft_weight_params(path) if self._is_ft_style else None
        )

        logging.debug(
            f"CkptDatabase all tensor names = {self.get_pretrain_tensor_names()}"
        )

    @property
    def is_ft_style(self) -> bool:
        return self._is_ft_style
    
    @property
    def is_safetensor(self) -> bool:
        return all(map(lambda file: file.is_safetensor(), self.pretrain_file_list))

    @property
    def ft_weight_params(self) -> Optional[Dict[str, Any]]:
        return self._ft_weight_params
    
    def get_max_file_size(self) -> int:
        return max([file.file_size for file in self.pretrain_file_list])

    def load_hf_meta(self, path: str):
        # avoid consolidated.safetensors in Mistral-Nemo-Instruct-2407
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            files = set(json.load(open(index))["weight_map"].values())
            for f in files:
                ckpt = CkptFileInfo(file_name=os.path.join(path, f))
                self.pretrain_file_list.append(ckpt)
            return

        # standard HF
        patterns = ["*.safetensors", "*.bin", "*.pth", "*.pt"]
        glob_files = {}

        for pattern in patterns:
            glob_files[pattern] = [file for file in Path(path).glob(pattern)]

        for _, value in glob_files.items():
            if len(value) != 0:
                exclude_pattern: re.Pattern[str] = re.compile(
                    r".*adapter_model\.bin.*|.*training_args\.bin.*"
                )
                for f in value:
                    if not exclude_pattern.match(f.name):
                        ckpt = CkptFileInfo(file_name=str(f))
                        self.pretrain_file_list.append(ckpt)
                break

    def load_ptuning_meta(self, ptuning_path: Optional[str]):
        if ptuning_path is None or not os.path.exists(ptuning_path):
            return
        for f in Path(ptuning_path).glob("pytorch_model.bin"):
            if not self._contains(f):
                ckpt = CkptFileInfo(
                    file_name=str(f), finetune_type=FinetuneType.ptuning
                )
                self.finetune_file_list.append(ckpt)

    def _contains(self, path: Path):
        for info in self.pretrain_file_list + self.finetune_file_list:
            if Path(info.file_name).resolve() == path.resolve():
                return True
        return False

    def get_pretrain_tensor_names(self) -> List[str]:
        tensor_names = []
        for ckptfile in self.pretrain_file_list:
            tensor_names.extend(ckptfile.get_tensor_names())

        for ckptfile in self.finetune_file_list:
            tensor_names.extend(ckptfile.get_tensor_names())

        return tensor_names

    def load_tensor(
        self, name: str, data_type: Optional[torch.dtype] = torch.float16
    ) -> List[torch.Tensor]:
        tensors = []
        for ckpt_file in self.pretrain_file_list:
            if name in ckpt_file.get_tensor_names():
                tensors.append(ckpt_file.load_tensor(name, data_type))

        for ckpt_file in self.finetune_file_list:
            if name in ckpt_file.get_tensor_names():
                tensors.append(ckpt_file.load_tensor(name, data_type))

        return tensors

    def get_tensor_type(self, name: str) -> torch.dtype:
        return self.pretrain_file_list[0].get_tensor_type(name)

    def get_tensor_order(self, name: str) -> List[int]:
        orders = []
        for ckpt_file in self.pretrain_file_list:
            if name in ckpt_file.get_tensor_names():
                orders.append(
                    (ckpt_file.file_name, ckpt_file.get_tensor_read_order(name))
                )

        for ckpt_file in self.finetune_file_list:
            if name in ckpt_file.get_tensor_names():
                orders.append(
                    (ckpt_file.file_name, ckpt_file.get_tensor_read_order(name))
                )

        return orders

    def load_tensors_by_prefix(
        self, prefix_list: List[str], device: str, direct_io: bool
    ) -> dict[str, List[torch.Tensor]]:
        try:
            from fast_safetensors import LoadWithShm
            loader = LoadWithShm(2 * 1024 * 1024 * 1024, device, direct_io)
            load_tensors = lambda ckptfile: loader.load_safetensors_to_device(ckptfile.file_name)
        except (ModuleNotFoundError, ImportError):
            load_tensors = lambda ckptfile: ckptfile.load_tensors(device, direct_io)

        res = {}
        for ckptfile in self.pretrain_file_list:
            if any(
                tensor.startswith(prefix_list) for tensor in ckptfile.get_tensor_names()
            ):
                tensors = load_tensors(ckptfile)
                for k, v in tensors.items():
                    if not k.startswith(prefix_list):
                        continue
                    if k not in res:
                        res[k] = [v]
                    else:
                        res[k].append(v)
        return res
    
    def fastsafetensors_weights_iterator(self, device: str, use_tqdm_on_load: bool):
        from fastsafetensors import ParallelLoader, SingleGroup
        def iterator(device: str, use_tqdm_on_load: bool):
            if torch.distributed.is_initialized():
                pg = torch.distributed.group.WORLD
            else:
                pg = SingleGroup()

            hf_weights_files = sorted(
                [file.file_name for file in self.pretrain_file_list]
            )
            if device == "cuda":
                device = f"cuda:{pg.rank()}"
                logging.debug(f"origin device is cuda, set to {device}")
            # Create loader
            iterator = ParallelLoader(
                pg,
                hf_weights_files=hf_weights_files,
                use_tqdm_on_load=use_tqdm_on_load,
                device=device,
                bbuf_size_kb=1024 * 1024 * 2,
                use_shm=True,
            )
            try:
                # Execute parallel iteration
                yield from iterator.iterate_weights()
            finally:
                iterator.loader.close()
        return iterator(device, use_tqdm_on_load)

    def get_lora_tensor_names(self, config_name: str) -> List[str]:
        return self.lora_ckpt.get_lora_tensor_names(config_name)

    def load_lora_tensor(
        self, lora_name: str, tensor_name: str, data_type: torch.dtype
    ) -> List[torch.Tensor]:
        return self.lora_ckpt.load_lora_tensor(lora_name, tensor_name, data_type)

    def load_lora(self, config_name: str, lora_path: str):
        self.lora_ckpt.load_lora(config_name, lora_path)

    def remove_lora(self, name: str):
        return self.lora_ckpt.remove_lora(name)

    def get_lora_config(self, config_name: str):
        return self.lora_ckpt.get_lora_config(config_name)

    def has_lora(self):
        return self.lora_ckpt.has_lora()

    def get_first_lora_name(self):
        return self.lora_ckpt.get_first_lora_name()

    def dump_lora_info(self) -> None:
        self.lora_ckpt.dump_lora_info()

    def _parse_weight_style(self, ckpt_path: str):
        if ckpt_path and os.path.exists(
            os.path.join(ckpt_path, "model.safetensors.index.json")
        ):
            meta_file = os.path.join(ckpt_path, "model.safetensors.index.json")
            logging.info(f"read weight style from: {meta_file}")
            with open(meta_file, "r") as reader:
                meta_json = json.loads(reader.read())
                return meta_json.get("is_ft_style_weight", False)
        else:
            return False

    def _parse_ft_weight_params(self, ckpt_path: str):
        meta_file = os.path.join(ckpt_path, "model.safetensors.index.json")
        with open(meta_file, "r") as reader:
            meta_json = json.loads(reader.read())
            return meta_json.get("__env__params__", None)
