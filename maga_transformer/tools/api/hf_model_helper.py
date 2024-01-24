import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple
from huggingface_hub import HfApi
from huggingface_hub.hf_api import ModelInfo

from maga_transformer.model_factory_register import ModelDict

class HfModelInfo:
    def __init__(self, origin_model_info: ModelInfo, model_config_file: Optional[str]):
        self.model_info = origin_model_info
        self.model_config = {}
        self.hf_local_dir = None
        if model_config_file:
            assert os.path.exists(model_config_file), f"path {model_config_file} is not exist, can't read config"
            with open(model_config_file, 'r') as f:
                self.model_config =  json.load(f)
            self.hf_local_dir = str(Path(model_config_file).parent)
        
    @property
    def param_count(self):
        if hasattr(self.model_info, 'safetensors') and self.model_info.safetensors != None:
            return self.model_info.safetensors.get("total", None)
        return None

    @property
    def ft_model_type(self) -> Optional[str]:
        if self.model_info is not None:
            ft_type = ModelDict.get_ft_model_type_by_hf_repo(self.model_info.modelId)
            if ft_type != None:
                return ft_type

        return ModelDict.get_ft_model_type_by_config(self.model_config)

class HfModelInfoHelper:
    _instance = None
    HF_URI_PREFIX = "https://huggingface.co/"

    def __init__(self):
        if HfModelInfoHelper._instance != None:
            raise Exception("This class is a singleton!")
        HfModelInfoHelper._instance = self        
        self.api = HfApi()
        # self._cached_model_info: List[ModelInfo] = [x for x in self.api.list_models()]  # too slow

    def get_hf_model_info(self, repo_or_link: str, revision: Optional[str] = None) -> HfModelInfo:
        model_info = self._get_model_info(repo_or_link, revision)
        model_config_file = self._get_model_config_file(repo_or_link, revision)

        hf_model_info = HfModelInfo(model_info, model_config_file)
        # import datetime
        # def dump_handler(obj):
        #     from huggingface_hub.hf_api import ModelInfo
        #     if isinstance(obj, datetime.datetime):
        #         return obj.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        #         #return obj.isoformat()
        # print(json.dumps(hf_model_info.model_info.__dict__, default=dump_handler))
        
        # json.dump(hf_model_info.model_info.__dict__, open(f"/home/luoli.hn/work/FasterTransformer/maga_transformer/tools/api/test/testdata/{repo_or_link.split('/')[-1]}/model_info.json", "w"), default=dump_handler)
        return hf_model_info
    
    def _get_model_info(self, repo_or_link: str, revision: Optional[str]) -> ModelInfo:
        repo = HfModelInfoHelper.get_repo_from_hf_link(repo_or_link)
        model_info = self.api.model_info(repo, revision=revision, timeout=10)
        return model_info

    def _get_model_config_file(self, repo_or_link: str, revision: Optional[str]):
        repo = HfModelInfoHelper.get_repo_from_hf_link(repo_or_link)
        if not self.api.file_exists(repo, "config.json", revision=revision):
            return None
        config_file = self.api.hf_hub_download(repo, "config.json", revision=revision)
        return config_file


    @staticmethod
    def is_from_hf(model_path: str):
        return model_path.startswith(HfModelInfoHelper.HF_URI_PREFIX)

    @staticmethod
    def get_instance() -> 'HfModelInfoHelper':
        if HfModelInfoHelper._instance == None:
            HfModelInfoHelper()
        return HfModelInfoHelper._instance

    @staticmethod
    def get_repo_from_hf_link(model_link: str):
        return model_link.replace(HfModelInfoHelper.HF_URI_PREFIX, "")

def get_model_info_from_hf(model_path_or_name: str, revision: Optional[str]= None) -> Tuple[str, str]:
    from huggingface_hub import snapshot_download
    if os.path.exists(model_path_or_name):
        info = HfModelInfo(None, os.path.join(model_path_or_name, 'config.json'))
    else:
        info = HF_MODEL_INFO_HELPER.get_hf_model_info(model_path_or_name, revision)

    if info.ft_model_type is None:
        raise Exception(f"failed to get type or type not supported in dir {model_path_or_name}")

    if not os.path.exists(model_path_or_name):
        logging.info(f"try download {model_path_or_name} from huggingface hub")
        local_path = snapshot_download(model_path_or_name, revision=revision)
    else:
        local_path = model_path_or_name

    logging.info(f"detected model type: {info.ft_model_type}, local_path: {local_path}")
    return local_path, info.ft_model_type

HF_MODEL_INFO_HELPER = HfModelInfoHelper.get_instance()
