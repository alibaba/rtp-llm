import os
import json
import logging
from typing import Optional, Tuple
from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo
from pathlib import Path

from maga_transformer.model_factory_register import ModelDict


class HfStyleModelInfo:
    HF_URI_PREFIX = "https://huggingface.co/"
    CONFIG_FILE = "config.json"
    MODEL_META_INFO_FILE_NAME = "model.safetensors.index.json"
    TORCH_BIN_INDEX_FILE_NAME = "pytorch_model.bin.index.json"
    api = HfApi()
    def __init__(self, repo_or_link: str, revision: Optional[str] = None):
        self._is_from_hf = self.is_from_hf(repo_or_link)
        self.model_info = None
        self.meta_info_file = None
        self.repo_or_link = repo_or_link
        if self._is_from_hf:
            self.model_info = self._get_model_info(repo_or_link, revision)
            self.model_config_file = self._get_model_config_file(repo_or_link, revision)
            self.meta_info_file = self._get_meta_info_file(repo_or_link, revision)
        else:
            self.model_info = None
            self.meta_info_file = None
            if os.path.exists(os.path.join(repo_or_link, HfStyleModelInfo.MODEL_META_INFO_FILE_NAME)):
                self.meta_info_file = os.path.join(repo_or_link, HfStyleModelInfo.MODEL_META_INFO_FILE_NAME)
            elif os.path.exists(os.path.join(repo_or_link, HfStyleModelInfo.TORCH_BIN_INDEX_FILE_NAME)):
                self.meta_info_file = os.path.join(repo_or_link, HfStyleModelInfo.TORCH_BIN_INDEX_FILE_NAME)
            self.model_config_file = os.path.join(repo_or_link, HfStyleModelInfo.CONFIG_FILE)

        self.hf_local_dir = os.path.dirname(self.model_config_file) if self.model_config_file else None

        # Load model information
        self.model_config = self._load_model_config(self.model_config_file)
        self.param_count, self.total_size = self._calculate_model_parameters()

        # Load auto config if available
        self.auto_config_py = self._get_auto_config_py(repo_or_link, revision) if self._is_from_hf else None

    def _get_model_info(self, repo_or_link: str, revision: Optional[str]) -> ModelInfo:
        repo = self._get_repo_from_hf_link(repo_or_link)
        return self.api.model_info(repo, revision=revision, timeout=10)

    def _get_model_config_file(self, repo_or_link: str, revision: Optional[str]):
        repo = self._get_repo_from_hf_link(repo_or_link)
        if self.api.file_exists(repo, self.CONFIG_FILE, revision=revision):
            return self.api.hf_hub_download(repo, self.CONFIG_FILE, revision=revision)
        return None

    def _get_meta_info_file(self, repo_or_link: str, revision: Optional[str]):
        repo = self._get_repo_from_hf_link(repo_or_link)
        if self.api.file_exists(repo, self.MODEL_META_INFO_FILE_NAME, revision=revision):
            return self.api.hf_hub_download(repo, self.MODEL_META_INFO_FILE_NAME, revision=revision)
        elif self.api.file_exists(repo, self.TORCH_BIN_INDEX_FILE_NAME, revision=revision):
            return self.api.hf_hub_download(repo, self.TORCH_BIN_INDEX_FILE_NAME, revision=revision)
        return None

    def _load_model_config(self, config_file: Optional[str]):
        logging.info(f'load config from {config_file}')
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}

    def _get_auto_config_py(self, repo_or_link: str, revision: Optional[str]):
        config_file = self.model_config_file
        repo = self._get_repo_from_hf_link(repo_or_link)

        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
                config_py = config_dict.get('auto_map', {}).get('AutoConfig')
                if config_py:
                    config_py_file = config_py.split('.')[0] + ".py" if config_py else None
                    return self.api.hf_hub_download(repo, config_py_file, revision=revision)
        return None

    def _calculate_model_parameters(self) -> Tuple[Optional[int], Optional[int]]:
        param_count = None
        total_size = None

        if self.model_info and self.model_info.safetensors:
            param_count = self.model_info.safetensors.total
            total_size = sum(count * 2 if weight_type in ['FP16', 'BF16', 'FP32', 'FP32', "INT8", 'F16']
                             else count for weight_type, count in self.model_info.safetensors.parameters.items())
        elif self.meta_info_file and os.path.exists(self.meta_info_file):
            logging.info(f'load meta_info from {self.meta_info_file}')
            with open(self.meta_info_file, 'r') as f:
                meta_info = json.load(f)
                total_size = meta_info.get("metadata", {}).get("total_size", None)
        if total_size is None and not self._is_from_hf:
            # try get file size from disk
                    # standard HF
            patterns = ["*.safetensors", "*.bin", "*.pth", "*.pt"]
            total_size = 0
            for pattern in patterns:
                for file in Path(self.repo_or_link).glob(pattern):
                    if os.path.isfile(file):
                        total_size += file.stat().st_size
            logging.info(f"fallback to get file size from disk: {total_size}")

        logging.info(f'{self.meta_info_file} {self.model_config_file} {self.model_info} param_count: {param_count}, total_size: {total_size}')
        return param_count, total_size

    @property
    def ft_model_type(self) -> Optional[str]:
        if self.model_info:
            # Assume ModelDict.get_ft_model_type_by_hf_repo() is a valid method
            ft_type = ModelDict.get_ft_model_type_by_hf_repo(self.model_info.modelId)
            if ft_type is not None:
                return ft_type
        return ModelDict.get_ft_model_type_by_config(self.model_config)

    @staticmethod
    def is_from_hf(model_path: str) -> bool:
        return model_path.startswith(HfStyleModelInfo.HF_URI_PREFIX) or not model_path.startswith(("oss:", "http:", "https:", "dfs:", "hdfs:", "/", "nas://"))

    @staticmethod
    def _get_repo_from_hf_link(model_link: str) -> str:
        return model_link.replace(HfStyleModelInfo.HF_URI_PREFIX, "")

def get_hf_model_info(model_path_or_name: str, revision: Optional[str] = None):
    info = HfStyleModelInfo(model_path_or_name, revision)
    return info

def get_model_info_from_hf(model_path_or_name: str, revision: Optional[str] = None) -> Tuple[str, str]:
    info = get_hf_model_info(model_path_or_name, revision)
    from huggingface_hub import snapshot_download
    
    if info.ft_model_type is None:
        raise Exception(f"failed to get type or type not supported in dir {model_path_or_name}")

    if not os.path.exists(model_path_or_name):
        logging.info(f"try download {model_path_or_name} from huggingface hub")
        local_path = snapshot_download(model_path_or_name, revision=revision)
    else:
        local_path = model_path_or_name

    logging.info(f"detected model type: {info.ft_model_type}, local_path: {local_path}")
    return local_path, info.ft_model_type

