import argparse
import fnmatch
import hashlib
import json
import logging
import multiprocessing
import os
import copy
import shutil
import time
from typing import Dict, Optional
import torch
from maga_transformer.utils.fuser import fetch_remote_file_to_local, MountRwMode
from maga_transformer.utils.time_util import timer_wrapper
from maga_transformer.utils.weight_type import WEIGHT_TYPE, get_weight_type_from_env
from maga_transformer.distribute.worker_info import ParallelInfo
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.model_factory import ModelFactory
from maga_transformer.models.base_model import ModelConfig
from maga_transformer.tools.api.model_basic_info_analyzer import parse_ft_model_type, parse_model_basic_info
from safetensors import safe_open


CUR_PATH: str = os.path.dirname(os.path.abspath(__file__))
ONE_MB = 1024 **2

class WeightConverter:
    def __init__(self, model_path: str, model_type: Optional[str], env_params: Dict[str,str]) -> None:
        self.model_basic_info = parse_model_basic_info(model_path, {})
        if self.model_basic_info is not None and not model_type:
            self.model_type = self.model_basic_info.ft_model_type
        elif model_type:
            self.model_type = model_type
        else:
            logging.error(f"not set model_type and cannot get model_type from {model_path}")
            raise RuntimeError("model_type is None")


        self.model_path: str | None = fetch_remote_file_to_local(model_path)
        self.env_params = env_params

        assert self.model_path
        if not model_type:
            model_type = parse_ft_model_type(self.model_path).get('ft_model_type', None)
            assert model_type
        self.model_type = model_type
        self.model_cls = ModelFactory.get_model_cls(self.model_type)


    def convert(self,  output_dir_base: str):
        output_dir_base = fetch_remote_file_to_local(output_dir_base, MountRwMode.RWMODE_RW)
        # 确定并发进程数，不超过tp_size
        pool_size = self._estimate_convert_parallel_num()
        logging.info(f"now start [{pool_size}] process tor convert")
        args_list = [
            (tp_rank, dp_rank, tp_rank + dp_rank * self.tp_size, output_dir_base)
                for dp_rank in range(self.dp_size)
                    for tp_rank in range(self.tp_size)
        ]
        logging.info(f'args : {args_list}')
        if pool_size > 1:
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(processes=pool_size) as pool:
                pool.starmap(self._convert, args_list)
        else:
            for (tp_rank, dp_rank, world_rank, _) in args_list:
                self._convert(tp_rank, dp_rank, world_rank, output_dir_base)
        # copy other files:
        self._save_converted(self.model_path, output_dir_base)

        return 0

    @property
    def tp_size(self):
        return int(self.env_params.get("TP_SIZE", "1"))

    @property
    def dp_size(self):
        return int(self.env_params.get("DP_SIZE", "1"))

    @property
    def world_size(self):
        return int(self.env_params.get("WORLD_SIZE", self.tp_size * self.dp_size))

    @staticmethod
    def get_free_mem_MB():
        import psutil
        memory_info = psutil.virtual_memory()
        free_memory = memory_info.free/ONE_MB
        return free_memory

    def _estimate_convert_parallel_num(self):
        max_pool_size = self._estimate_max_convert_parallel_num()
        return self.world_size if max_pool_size > self.world_size else max_pool_size

    def _estimate_max_convert_parallel_num(self):
        converter_num_per_gpu = int(os.environ.get("CONVERTER_NUM_PER_GPU", "4"))
        try:
            cuda_count = torch.cuda.device_count()
            assert(cuda_count >= 1)
            return cuda_count * converter_num_per_gpu
        except Exception as _:
            logging.info("no cuda device convert by cpu")
            free_mb = self.get_free_mem_MB() * 0.8
            dump_buffer_size_mb = 10 * 1024  # 10G dump once
            if self.model_basic_info.model_size:
                model_size_mb = self.model_basic_info.model_size/ONE_MB

                env_params = copy.deepcopy(self.env_params)
                model_config = ModelConfig(
                    model_type=self.model_type,
                    ckpt_path=self.model_path,
                    weight_type=get_weight_type_from_env(env_params),
                    act_type=WEIGHT_TYPE.from_str(env_params.get("ACT_TYPE", "FP16")),
                    ptuning_path=None,
                    max_seq_len=0,
                    tokenizer_path=self.model_path
                )
                paralle_info = ParallelInfo.from_params(env_params)
                config: GptInitModelParameters = self.model_cls.create_config(model_config, paralle_info)

                one_layer_model_size_mb = model_size_mb/config.layer_num
                if model_size_mb < dump_buffer_size_mb:
                    need_size_mb = model_size_mb
                else:
                    need_size_mb = dump_buffer_size_mb + one_layer_model_size_mb
                if free_mb // need_size_mb > self.tp_size:
                    return int(free_mb // need_size_mb)
                else:
                    return int(free_mb // need_size_mb if free_mb // need_size_mb > 1 else 1)
            return 1

    @timer_wrapper('convert 1 tp')
    def _convert(self, tp_rank:int, dp_rank:int, world_rank: int, output_dir_base: str):
        env_params = copy.deepcopy(self.env_params)
        for env_key, env_value in env_params.items():
            os.environ[env_key] = env_value
        try:
            cuda_device_list = [str(i) for i in range(torch.cuda.device_count())]
            if len(cuda_device_list) > 0:
                env_params.update({"LOCAL_WORLD_SIZE" : min(len(cuda_device_list), self.world_size)})
        except Exception as _:
            logging.info(f"no GPU device, load to mem")
        env_params.update({"WORLD_RANK": world_rank})
        env_params.update({"DP_RANK": dp_rank})
        env_params.update({"TP_RANK": tp_rank})

        model_config = ModelConfig(
            model_type=self.model_type,
            ckpt_path=self.model_path,
            weight_type=get_weight_type_from_env(env_params),
            act_type=WEIGHT_TYPE.from_str(env_params.get("ACT_TYPE", "FP16")),
            ptuning_path=None,
            max_seq_len=0,
            tokenizer_path=self.model_path
        )
        paralle_info = ParallelInfo.from_params(env_params)
        logging.info(f"begin convert model rank:{paralle_info}")
        config: GptInitModelParameters = self.model_cls.create_config(model_config, paralle_info)
        model = self.model_cls(config)
        loader = model.create_model_loader(paralle_info)
        max_retry_times = 3
        for i in range(max_retry_times):
            try:
                loader.dump_weight_as_ft_style(paralle_info.device, output_dir_base)
                logging.info(f"dump rank:[{world_rank}] done")
                break
            except Exception as e:
                logging.warn(f"dump rank:[{world_rank}] failed, {str(e)}, retry {i} times")
                if i == max_retry_times -1:
                    logging.error(f"dump rank:[{world_rank}] retry {i} times, but still failed")
                    raise RuntimeError(f"Failed after 10 retries: {str(e)}") from e
                continue
        logging.info(f"convert model rank:{world_rank} done")


    @timer_wrapper('save convert result')
    def _save_converted(self, input_path, output_path: str):
        self._copy_filtered_files(input_path, output_path, ['pytorch_model.bin.index.json', 'model.safetensors.index.json', '*.safetensors', '*.bin', '*.pth', '*.pt', '*.gguf'])
        self._generate_safetensor_meta_info(output_path)
        # touch done
        done_file = os.path.join(output_path, 'done')
        with open(done_file, 'w') as f:
            pass

    @timer_wrapper('generate safetensors meta info')
    def _generate_safetensor_meta_info(self, output_path: str):
        weight_map = {}
        total_size = 0

        # 获取所有 .safetensor 文件，并按名称排序
        safetensor_files = sorted([
            f for f in os.listdir(output_path)
            if f.endswith('.safetensors')
        ])

        if not safetensor_files:
            logging.info("指定目录下没有找到任何 .safetensors 文件。")
            return -1

        for st_file in safetensor_files:
            st_path = os.path.join(output_path, st_file)
            total_size += os.path.getsize(st_path)

            # 使用 safetensors 库打开文件并读取所有键（权重名称）
            try:
                with safe_open(st_path, framework="pt") as f:
                    keys = f.keys()
                    for key in keys:
                        if key in weight_map:
                            logging.warning(f"警告: 权重 '{key}' 已经映射到 '{weight_map[key]}', 现在尝试映射到 '{st_file}'。")
                        weight_map[key] = st_file
            except Exception as e:
                logging.warning(f"无法读取文件 '{st_file}': {e}")

        index = {
            "metadata": {
                "total_size": total_size
            },
            "weight_map": weight_map,
            "is_ft_style_weight": True,
            "__env__params__": self.env_params
        }

        # 将索引数据写入 JSON 文件
        output_file = 'model.safetensors.index.json'
        output_path = os.path.join(output_path, output_file)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=4, ensure_ascii=False)
            logging.info(f"索引文件已成功生成: {output_file}")
        except Exception as e:
            logging.info(f"写入索引文件失败: {e}")
            return -1

    @staticmethod
    def _copy_filtered_files(src_dir, dst_dir, exclude_patterns):
        """
        复制src_dir目录下的所有文件到dst_dir目录，过滤掉匹配exclude_patterns模式的文件。

        参数：
        - src_dir: 源目录路径。
        - dst_dir: 目标目录路径。
        - exclude_patterns: 要过滤的文件模式列表，例如["*.safetensors", "*.bin", "*.pth", "*.pt"]。
        """
        for root, dirs, files in os.walk(src_dir):
            # 计算当前目录相对于源目录的相对路径
            rel_path = os.path.relpath(root, src_dir)
            # 构建目标目录的路径
            dst_path = os.path.join(dst_dir, rel_path)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
                logging.info(f"create dir:{dst_path}")
            for file in files:
                # 检查文件是否匹配任何一个排除模式
                exclude = False
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(file, pattern):
                        exclude = True
                        logging.info(f"exclude file:{file}")
                        break
                if not exclude:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(dst_path, file)
                    logging.info(f"copy file:{src_file} to {dst_file}")
                    shutil.copy2(src_file, dst_file)  # 复制文件，保留元数据

def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='convert model weights to ft_style_weight.')

    # 添加参数
    parser.add_argument('--pretrained_model_dir', type=str, help='Pretrained model path')
    parser.add_argument('--output_dir_base', type=str, help='Output base folder')
    parser.add_argument('--model_type', type=str, default= '', help='[Optinal] the model_type to be convert.')
    parser.add_argument('--env_params', type=str, default='{}', help='[Optinal] env args.')

    # 解析参数
    args = parser.parse_args()
    converter = WeightConverter(args.pretrained_model_dir, args.model_type, json.loads(args.env_params))

    ret_code = converter.convert(args.output_dir_base)
    exit(ret_code)


if __name__ == '__main__':
    # logging.config.dictConfig(LOGGING_CONFIG)
    main()
