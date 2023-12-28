import json
import os
from pathlib import Path

from maga_transformer.tools.api.model_dict import ModelDict
class HfModelInfo:
    def __init__(self, origin_model_info, model_config_file):
        self.model_info = origin_model_info
        self.model_config = {}
        self.hf_local_dir = None
        if model_config_file:
            with open(model_config_file, 'r') as f:
                self.model_config =  json.load(f)
            self.hf_local_dir = str(Path(model_config_file).parent)
        
    @property
    def param_count(self):
        if hasattr(self.model_info, 'safetensors') and self.model_info.safetensors != None:
            return self.model_info.safetensors.get("total", None)
        return None

    @property
    def ft_model_type(self):
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
        os.environ.setdefault("HF_ENDPOINT", "https://whale-hf-mirror.alibaba-inc.com")
        from huggingface_hub import HfApi
        self.api = HfApi()
        # self._cached_model_info: List[ModelInfo] = [x for x in self.api.list_models()]  # too slow

    def get_hf_model_info(self, repo_or_link: str):
        repo = HfModelInfoHelper.get_repo_from_hf_link(repo_or_link)
        model_info = HF_MODEL_INFO_HELPER._get_model_info(repo)
        model_config_file = HF_MODEL_INFO_HELPER._get_model_config_file(repo)

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
    
    def _get_model_info(self, repo_or_link) -> 'ModelInfo':
        repo = HfModelInfoHelper.get_repo_from_hf_link(repo_or_link)
        model_info = self.api.model_info(repo)
        return model_info

    def _get_model_config_file(self, repo_or_link, download_dir=None):
        repo = HfModelInfoHelper.get_repo_from_hf_link(repo_or_link)
        if not self.api.file_exists(repo, "config.json"):
            return None
        config_file = self.api.hf_hub_download(repo, "config.json")
        return config_file


    @staticmethod
    def is_from_hf(model_path):
        return model_path.startswith(HfModelInfoHelper.HF_URI_PREFIX)

    @staticmethod
    def get_instance():
        if HfModelInfoHelper._instance == None:
            HfModelInfoHelper()
        return HfModelInfoHelper._instance

    @staticmethod
    def get_repo_from_hf_link(model_link):
        return model_link.replace(HfModelInfoHelper.HF_URI_PREFIX, "")



HF_MODEL_INFO_HELPER = HfModelInfoHelper.get_instance()
