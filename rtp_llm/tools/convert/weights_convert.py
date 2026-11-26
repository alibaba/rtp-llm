import argparse
import copy
import fnmatch
import json
import logging
import multiprocessing
import os
import shutil
from typing import Dict, Optional

import torch

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig, build_model_config
from rtp_llm.config.py_config_modules import (
    MIN_WORKER_INFO_PORT_NUM,
    QuantizationConfig,
    VitConfig,
)
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.models_py.distributed.rank_layout import RankLayout
from rtp_llm.ops import (
    FMHAConfig,
    HWKernelConfig,
    MoeConfig,
    ParallelismConfig,
    ProfilingDebugLoggingConfig,
    TaskType,
)
from rtp_llm.tools.api.hf_model_helper import HfStyleModelInfo
from rtp_llm.tools.convert.ft_export_job import DraftFTDescriptor, create_ft_export_job
from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index
from rtp_llm.tools.convert.weights_convert_utils import (
    apply_layer_override_and_post_build,
)
from rtp_llm.utils.fuser import MountRwMode, fetch_remote_file_to_local
from rtp_llm.utils.time_util import timer_wrapper

CUR_PATH: str = os.path.dirname(os.path.abspath(__file__))
ONE_MB = 1024**2


class WeightConverter:
    def __init__(
        self, model_path: str, model_type: Optional[str], env_params: Dict[str, str]
    ) -> None:
        self.model_path = os.path.abspath(fetch_remote_file_to_local(model_path))
        self.env_params = env_params
        # The worker budget needs checkpoint bytes, not a serving model's
        # runtime/KV cache size. Read the mounted checkpoint's own metadata.
        checkpoint = HfStyleModelInfo(self.model_path)
        self.checkpoint_size = checkpoint.total_size
        if not model_type:
            model_type = checkpoint.ft_model_type
            if not model_type:
                raw = checkpoint.model_config
                if raw.get("architectures") == ["LlamaForCausalLMEagle3"]:
                    # This architecture also labels incompatible reduced-vocab
                    # drafts. Accept only the serving adapter's exact contract.
                    from rtp_llm.models_py.new_models.qwen3_moe_eagle3.proposal import (
                        Eagle3FullVocabularyDraft,
                    )

                    Eagle3FullVocabularyDraft.capture_ids(raw)
                    model_type = "qwen_3_moe_eagle3"
            if not model_type:
                raise ValueError("Cannot infer FT model type; specify --model_type")
        self.model_type = model_type
        self.model_cls = ModelFactory.get_model_cls(self.model_type)

    def convert(self, output_dir_base: str):
        output_dir_base = fetch_remote_file_to_local(
            output_dir_base, MountRwMode.RWMODE_RW
        )
        if os.path.commonpath(
            (os.path.realpath(self.model_path), os.path.realpath(output_dir_base))
        ) == os.path.realpath(self.model_path):
            raise ValueError("FT output must not be inside the source checkpoint")
        # A previous index, shard, or completion marker must never be attributed
        # to this conversion, including when a job fails partway through.
        if os.path.isdir(output_dir_base) and any(
            name == "done" or name == "model.safetensors.index.json"
            or name.endswith((".safetensors", "-preshuffled.json")) for name in os.listdir(output_dir_base)
        ):
            raise FileExistsError("FT destination already contains a checkpoint; use a fresh directory")
        # 确定并发进程数，不超过tp_size
        pool_size = self._estimate_convert_parallel_num()
        logging.info(f"now start [{pool_size}] process tor convert")
        args_list = [
            (tp_rank, dp_rank, tp_rank + dp_rank * self.tp_size, output_dir_base)
            for dp_rank in range(self.dp_size)
            for tp_rank in range(self.tp_size)
        ]
        logging.info(f"args : {args_list}")
        if pool_size > 1:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=pool_size) as pool:
                ranks = pool.starmap(self._convert, args_list)
        else:
            ranks = [self._convert(*args) for args in args_list]
        # copy other files:
        self._save_converted(self.model_path, output_dir_base, ranks)

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
        free_memory = memory_info.free / ONE_MB
        return free_memory

    def _estimate_convert_parallel_num(self):
        max_pool_size = self._estimate_max_convert_parallel_num()
        return self.world_size if max_pool_size > self.world_size else max_pool_size

    def _estimate_max_convert_parallel_num(self):
        # Get converter_num_per_gpu from environment variable, default to 4
        converter_num_per_gpu = int(os.environ.get("CONVERTER_NUM_PER_GPU", "4"))
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if cuda_count:
            return cuda_count * converter_num_per_gpu
        logging.info("no CUDA device; converting on CPU")
        model_size = self.checkpoint_size
        if not model_size:
            return 1
        model_size_mb = model_size / ONE_MB
        dump_buffer_mb = 10 * 1024
        need_mb = model_size_mb
        if model_size_mb >= dump_buffer_mb:
            config = self.model_cls._create_config(self.model_path)
            need_mb = dump_buffer_mb + model_size_mb / max(1, config.num_layers)
        return max(1, int(self.get_free_mem_MB() * 0.8 // need_mb))

    def _build_parallelism_config(self) -> ParallelismConfig:
        """Build parallelism_config by reading env vars directly (TP_SIZE, WORLD_SIZE, etc.)."""

        def _env_int(key: str, default: int) -> int:
            v = os.environ.get(key)
            return int(v) if v is not None else default

        pc = ParallelismConfig()
        pc.tp_size = _env_int("TP_SIZE", 1)
        pc.dp_size = _env_int("DP_SIZE", 1)
        pc.pp_size = 1
        pc.world_size = _env_int("WORLD_SIZE", pc.tp_size * pc.dp_size)
        pc.world_rank = _env_int("WORLD_RANK", 0)
        pc.local_world_size = _env_int("LOCAL_WORLD_SIZE", 1)
        pc.ep_size = _env_int("EP_SIZE", 0)

        if pc.world_size > 1 and pc.local_world_size == 1:
            n = (
                torch.cuda.device_count()
                if torch.cuda.is_available()
                else pc.world_size
            )
            pc.local_world_size = max(min(n, pc.world_size), 1)
        if pc.ep_size == 1:
            assert (
                pc.tp_size >= 1
            ), f"Pure TP mode (ep_size=1) requires tp_size >= 1, got tp_size={pc.tp_size}"
            assert (
                pc.dp_size == 1
            ), f"Pure TP mode (ep_size=1) requires dp_size == 1, got dp_size={pc.dp_size}"
        elif pc.ep_size == 0:
            logging.info("ep_size == 0, auto set to world size")
            pc.ep_size = pc.tp_size * pc.dp_size
        else:
            assert pc.ep_size == pc.tp_size * pc.dp_size, (
                f"ep_size must be equal to 1 or tp_size * dp_size, got ep_size={pc.ep_size}, "
                f"tp_size={pc.tp_size}, dp_size={pc.dp_size}"
            )

        layout = RankLayout.from_parallelism_config(pc)
        coord = layout.coord_of(pc.world_rank)
        pc.tp_rank = _env_int("TP_RANK", coord.tp)
        pc.dp_rank = _env_int("DP_RANK", coord.dp)
        pc.ep_rank = layout.ep_rank_of(pc.world_rank, pc.ep_size)
        pc.local_rank = pc.world_rank % pc.local_world_size
        pc.ffn_tp_size = pc.tp_size
        pc.ffn_tp_rank = pc.tp_rank % pc.ffn_tp_size if pc.ffn_tp_size else 0
        return pc

    @timer_wrapper("convert 1 tp")
    def _convert(
        self, tp_rank: int, dp_rank: int, world_rank: int, output_dir_base: str
    ):
        env_params = copy.deepcopy(self.env_params)
        # Set rank in env first so _build_parallelism_config() sees them (it reads os.environ)
        env_params["WORLD_RANK"] = world_rank
        env_params["DP_RANK"] = dp_rank
        env_params["TP_RANK"] = tp_rank
        for env_key, env_value in env_params.items():
            os.environ[env_key] = str(env_value)
        try:
            cuda_device_list = [str(i) for i in range(torch.cuda.device_count())]
            if len(cuda_device_list) > 0:
                env_params.update(
                    {"LOCAL_WORLD_SIZE": min(len(cuda_device_list), self.world_size)}
                )
                os.environ["LOCAL_WORLD_SIZE"] = str(env_params["LOCAL_WORLD_SIZE"])
        except Exception as _:
            logging.info(f"no GPU device, load to mem")

        # Get quantization from env_params (compatibility logic)
        quantization = env_params.get("QUANTIZATION", "")
        if not quantization:
            int8_mode = env_params.get("INT8_MODE", "0")
            weight_type = env_params.get("WEIGHT_TYPE", "").upper()
            if int(int8_mode) == 1 or weight_type == "INT8":
                quantization = "INT8"

        # Create config using _create_config
        model_config: ModelConfig = self.model_cls._create_config(self.model_path)

        # Create ModelArgs from config
        model_args = ModelArgs()
        model_args.ckpt_path = self.model_path
        model_args.tokenizer_path = self.model_path
        model_args.model_type = self.model_type
        model_args.act_type = env_params.get("ACT_TYPE", "")

        kv_cache_config = KVCacheConfig()
        kv_cache_config.seq_size_per_block = 64
        kv_cache_config.fp8_kv_cache = int(env_params.get("FP8_KV_CACHE", 0))

        quantization_config = QuantizationConfig()
        quantization_config.quantization = quantization
        if "W8A8_QUANT_CHUNK_ROWS" in env_params:
            quantization_config.w8a8_quant_chunk_rows = int(env_params["W8A8_QUANT_CHUNK_ROWS"])

        # Build model config
        build_model_config(
            model_config=model_config,
            model_args=model_args,
            kv_cache_config=kv_cache_config,
            quantization_config=quantization_config,
            profiling_debug_logging_config=ProfilingDebugLoggingConfig(),
            embedding_config=None,  # Fake loader doesn't need embedding_config
        )
        apply_layer_override_and_post_build(model_config, self.model_cls, env_params)

        parallelism_config = self._build_parallelism_config()

        hw_kernel_config = HWKernelConfig()
        if (
            self.model_type in (
                "qwen_3_moe_eagle3", "qwen_3_dflash", "qwen_2-mtp", "qwen_3_dspark"
            )
            and model_config.task_type == TaskType.LANGUAGE_MODEL
            and not model_config.enable_output_vocab_pruning
        ):
            model = DraftFTDescriptor(
                model_config=model_config,
                parallelism_config=parallelism_config,
                hw_kernel_config=hw_kernel_config,
                kv_cache_config=kv_cache_config,
                get_weight_cls=self.model_cls.get_weight_cls,
            )
        else:
            # Ordinary models still need their tokenizer's task/vocab metadata.
            # Construct the unloaded descriptor, not a serving model.
            model = self.model_cls(
                model_config=model_config,
                parallelism_config=parallelism_config,
                hw_kernel_config=hw_kernel_config,
                kv_cache_config=kv_cache_config,
                fmha_config=FMHAConfig(),
                moe_config=MoeConfig(),
                load_method=LoadMethod.AUTO,
                max_generate_batch_size=0,
                vit_config=VitConfig(),
                merge_lora=False,
            )
        writer = create_ft_export_job(model)
        device_str = (
            f"cuda:{parallelism_config.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        max_retry_times = 3
        for i in range(max_retry_times):
            try:
                writer.dump_weight_as_ft_style(device_str, output_dir_base)
                logging.info(f"dump rank:[{world_rank}] done")
                break
            except Exception as e:
                logging.warn(
                    f"dump rank:[{world_rank}] failed, {str(e)}, retry {i} times"
                )
                if i == max_retry_times - 1:
                    logging.error(
                        f"dump rank:[{world_rank}] retry {i} times, but still failed"
                    )
                    raise RuntimeError(f"FT export failed after {max_retry_times} attempts") from e
                continue
        logging.info(f"convert model rank:{world_rank} done")
        return f"rank_{tp_rank:02d}_{dp_rank:02d}_{parallelism_config.ep_rank:02d}"

    @timer_wrapper("save convert result")
    def _save_converted(self, input_path, output_path: str, ranks):
        self._copy_filtered_files(
            input_path,
            output_path,
            [
                "pytorch_model.bin.index.json",
                "model.safetensors.index.json",
                "*-preshuffled.json",
                "*.safetensors",
                "*.bin",
                "*.pth",
                "*.pt",
                "*.gguf",
                "done",
            ],
        )
        if self.model_type == "deepseek_v41":
            from rtp_llm.tools.convert.ft_export_engram import export_v41_engram

            export_v41_engram(input_path, output_path)
        self._generate_safetensor_meta_info(output_path, ranks)
        # touch done
        done_file = os.path.join(output_path, "done")
        with open(done_file, "w") as f:
            pass

    def _generate_safetensor_meta_info(self, output_path, expected_ranks=None):
        generate_ft_index(output_path, self.env_params, expected_ranks=expected_ranks)

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
    parser = argparse.ArgumentParser(
        description="convert model weights to ft_style_weight."
    )

    # 添加参数
    parser.add_argument(
        "--pretrained_model_dir", type=str, help="Pretrained model path"
    )
    parser.add_argument("--output_dir_base", type=str, help="Output base folder")
    parser.add_argument(
        "--model_type",
        type=str,
        default="",
        help="[Optinal] the model_type to be convert.",
    )
    parser.add_argument(
        "--env_params", type=str, default="{}", help="[Optinal] env args."
    )

    # 解析参数
    args = parser.parse_args()
    converter = WeightConverter(
        args.pretrained_model_dir, args.model_type, json.loads(args.env_params)
    )

    ret_code = converter.convert(args.output_dir_base)
    exit(ret_code)


if __name__ == "__main__":
    # logging.config.dictConfig(LOGGING_CONFIG)
    main()
