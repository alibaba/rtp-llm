from enum import Enum
import os
import logging
from typing import Any, Dict, NamedTuple, Optional
import datasets
from urllib.parse import urlparse
from maga_transformer.utils.fuser import fetch_remote_file_to_local
from maga_transformer.utils.time_util import Timer, timer_wrapper

class DatasetParams(NamedTuple):
    source: str
    data_format: Optional[str] = None
    load_args: Dict[str, Any] = {}

class DatasetType(Enum):
    RTP_LLM_ACCESS_LOG = 1
    RTP_LLM_ACCESS_LOG_JSON_STR = 2
    TEXT = 3
    CHAT_PROMPT = 4

    @classmethod
    def from_str(cls, value: Optional[str]) -> 'DatasetType':
        lower_value = value.lower() if value else None
        for name, member in cls.__members__.items():
            if lower_value == name.lower():
                return member
        raise ValueError('No enum member with value %s' % value)


class DatasetsAdapter:
    @staticmethod
    @timer_wrapper('load dataset')
    def load_dataset(dataset_params: DatasetParams) -> datasets.Dataset:
        # 如果是文件从文件load
        source = dataset_params.source
        data_format = dataset_params.data_format
        parse_result = urlparse(source)
        if parse_result.scheme:
            # from remote, fetch remote to local
            local_path = fetch_remote_file_to_local(source)
            return DatasetsAdapter._load_dataset_from_local(local_path, data_format, **dataset_params.load_args)
        elif os.path.exists(source):
            return DatasetsAdapter._load_dataset_from_local(source, data_format, **dataset_params.load_args)
        # 否则从已经存在的dataset中获取
        return datasets.load_dataset(source, **dataset_params.load_args)

    @staticmethod
    def _load_dataset_from_local(path: str, data_format: Optional[str] = None, **load_args):
        if os.path.isdir(path):
            assert len(os.listdir(path)) == 1
            path = os.path.join(path, os.listdir(path)[0])
            
        if not data_format:
            if path.endswith(('.json', '.jsonl')):
                data_format = 'json'
            elif path.endswith('.csv'):
                data_format = 'csv'
            elif path.endswith('.text'):
                data_format = 'text'
            elif path.endswith('.pkl'):
                data_format = 'pandas'
            elif path.endswith('access.log') or path.split("/")[-1].startswith("access.log-"):
                data_format = 'text'
            else:
                data_format = 'text'
        return datasets.load_dataset(data_format.lower(), data_files=path, **load_args)


    @staticmethod
    def parse_dataset_type(dataset: datasets.Dataset, dataset_params: DatasetParams) -> Optional[DatasetType]:
        if set(['log_time', 'request.request_json', 'response.responses']).issubset(set(dataset.column_names)):
            return DatasetType.RTP_LLM_ACCESS_LOG
        elif 'prompt' in dataset.column_names:
            return DatasetType.CHAT_PROMPT
        elif 'text' in dataset.column_names:
            dataset_type = DatasetsAdapter.parse_dataset_type_from_dataset_params(dataset_params)
            return dataset_type if dataset_type else DatasetType.TEXT 
        return DatasetType.from_str(dataset_params.data_format)

    @staticmethod
    def parse_dataset_type_from_dataset_params(dataset_params: DatasetParams):
        source = dataset_params.source
        parse_result = urlparse(source)
        local_path: Optional[str] = None 
        if parse_result.scheme:
            # from remote, fetch remote to local
            local_path = fetch_remote_file_to_local(source)
        elif os.path.exists(source):
            local_path = source
        else:
            return None
        if os.path.isdir(local_path):
            assert len(os.listdir(local_path)) == 1
            local_path = os.listdir(local_path)[0]
        if local_path.endswith('access.log') or local_path.split("/")[-1].startswith("access.log-"):
            return DatasetType.RTP_LLM_ACCESS_LOG_JSON_STR

        
