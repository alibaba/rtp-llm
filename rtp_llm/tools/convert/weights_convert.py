import argparse
import copy
import fnmatch
import json
import logging
import multiprocessing
import os
import shutil
import sys
from typing import Dict, Optional

import torch
from safetensors import safe_open

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
from rtp_llm.ops import (
    DeviceResourceConfig,
    FMHAConfig,
    HWKernelConfig,
    MoeConfig,
    ParallelismConfig,
    ProfilingDebugLoggingConfig,
)
from rtp_llm.server.server_args.util import str2_cp_rotate_method
from rtp_llm.tools.api.model_basic_info_analyzer import (
    parse_ft_model_type,
    parse_model_basic_info,
)
from rtp_llm.utils.fuser import MountRwMode, fetch_remote_file_to_local
from rtp_llm.utils.time_util import timer_wrapper

CUR_PATH: str = os.path.dirname(os.path.abspath(__file__))
ONE_MB = 1024**2


class WeightConverter:
    def __init__(
        self,
        model_path: str,
        model_type: Optional[str],
        env_params: Dict[str, str],
        draft_model_type: Optional[str] = None,
        draft_model_path: Optional[str] = None,
    ) -> None:
        self.model_basic_info = parse_model_basic_info(model_path, {})
        if self.model_basic_info is not None and not model_type:
            self.model_type = self.model_basic_info.ft_model_type
        elif model_type:
            self.model_type = model_type
        else:
            logging.error(
                f"not set model_type and cannot get model_type from {model_path}"
            )
            raise RuntimeError("model_type is None")

        self.model_path: str | None = fetch_remote_file_to_local(model_path)
        self.env_params = env_params

        assert self.model_path
        if not model_type:
            model_type = parse_ft_model_type(self.model_path).get("ft_model_type", None)
            assert model_type
        self.model_type = model_type
        self.model_cls = ModelFactory.get_model_cls(self.model_type)
        # A speculative draft model lives in the same checkpoint as its target, so
        # it has to be dumped alongside it; otherwise serving with speculative
        # decoding falls back to reading the raw checkpoint and the ft_style dump
        # is useless to it.
        self.draft_model_type = (
            draft_model_type or env_params.get("SP_MODEL_TYPE") or ""
        )
        self.draft_model_path = draft_model_path or env_params.get("DRAFT_SRC") or ""
        if self.draft_model_type:
            ModelFactory.get_model_cls(self.draft_model_type)

    def _convert_scope(self) -> str:
        return str(self.env_params.get("CONVERT_SCOPE", "both")).lower()

    def convert(self, output_dir_base: str):
        output_dir_base = fetch_remote_file_to_local(
            output_dir_base, MountRwMode.RWMODE_RW
        )
        # 确定并发进程数，不超过tp_size
        pool_size = self._estimate_convert_parallel_num()
        logging.info(f"now start [{pool_size}] process tor convert")
        scope = self._convert_scope()
        if scope == "draft":
            if not self.draft_model_type:
                raise ValueError("CONVERT_SCOPE=draft requires SP_MODEL_TYPE")
            model_types = [self.draft_model_type]
        elif scope == "target":
            model_types = [self.model_type]
        else:
            model_types = [self.model_type]
            if self.draft_model_type:
                model_types.append(self.draft_model_type)
        logging.info(f"convert scope={scope}, model_types={model_types}")
        args_list = [
            (
                tp_rank,
                dp_rank,
                tp_rank + dp_rank * self.tp_size,
                output_dir_base,
                model_type,
            )
            for model_type in model_types
            for dp_rank in range(self.dp_size)
            for tp_rank in range(self.tp_size)
        ]
        logging.info(f"args : {args_list}")
        if pool_size > 1:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=pool_size) as pool:
                pool.starmap(self._convert, args_list)
        else:
            for tp_rank, dp_rank, world_rank, _, model_type in args_list:
                self._convert(tp_rank, dp_rank, world_rank, output_dir_base, model_type)
        # copy other files:
        meta_src = self.model_path
        if scope == "draft" and self.draft_model_path:
            meta_src = self.draft_model_path
        self._save_converted(meta_src, output_dir_base)

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
        try:
            cuda_count = torch.cuda.device_count()
            assert cuda_count >= 1
            return cuda_count * converter_num_per_gpu
        except Exception as _:
            logging.info("no cuda device convert by cpu")
            free_mb = self.get_free_mem_MB() * 0.8
            dump_buffer_size_mb = 10 * 1024  # 10G dump once
            if self.model_basic_info.model_size:
                model_size_mb = self.model_basic_info.model_size / ONE_MB

                env_params = copy.deepcopy(self.env_params)
                # Get quantization from env_params (compatibility logic)
                quantization = env_params.get("QUANTIZATION", "")
                if not quantization:
                    int8_mode = env_params.get("INT8_MODE", "0")
                    weight_type = env_params.get("WEIGHT_TYPE", "").upper()
                    if int(int8_mode) == 1 or weight_type == "INT8":
                        quantization = "INT8"
                config: ModelConfig = self.model_cls._create_config(self.model_path)
                # Apply settings from env_params
                config.model_type = self.model_type
                config.ckpt_path = self.model_path
                config.tokenizer_path = self.model_path
                config.act_type = env_params.get("ACT_TYPE", "")
                config.quantization = quantization
                config.max_seq_len = 0

                one_layer_model_size_mb = model_size_mb / config.num_layers
                if model_size_mb < dump_buffer_size_mb:
                    need_size_mb = model_size_mb
                else:
                    need_size_mb = dump_buffer_size_mb + one_layer_model_size_mb
                if free_mb // need_size_mb > self.tp_size:
                    return int(free_mb // need_size_mb)
                else:
                    return int(
                        free_mb // need_size_mb if free_mb // need_size_mb > 1 else 1
                    )
            return 1

    def _build_parallelism_config(self) -> ParallelismConfig:
        """Build parallelism_config from --env_params, falling back to the ambient env."""

        def _lookup(key: str) -> Optional[str]:
            v = self.env_params.get(key)
            return str(v) if v is not None else os.environ.get(key)

        def _env_int(key: str, default: int) -> int:
            v = _lookup(key)
            return int(v) if v is not None else default

        pc = ParallelismConfig()
        pc.tp_size = _env_int("TP_SIZE", 1)
        pc.dp_size = _env_int("DP_SIZE", 1)
        pc.pp_size = 1
        pc.world_size = _env_int("WORLD_SIZE", 1)
        pc.world_rank = _env_int("WORLD_RANK", 0)
        pc.local_world_size = _env_int("LOCAL_WORLD_SIZE", 1)
        pc.ep_size = _env_int("EP_SIZE", 0)
        pc.ffn_sp_size = _env_int("FFN_SP_SIZE", 1)

        cp_rotate_method = _lookup("CP_ROTATE_METHOD")
        if cp_rotate_method:
            pc.prefill_cp_config.method = str2_cp_rotate_method(cp_rotate_method)
            pc.prefill_cp_config.kv_cache_sharded = (
                _lookup("CP_KV_CACHE_SHARDED") or "0"
            ) == "1"

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

        pc.tp_rank = _env_int("TP_RANK", pc.world_rank % pc.tp_size)
        pc.dp_rank = _env_int("DP_RANK", pc.world_rank // pc.tp_size)
        pc.ep_rank = pc.world_rank % pc.ep_size
        pc.local_rank = pc.world_rank % pc.local_world_size
        pc.ffn_tp_size = pc.tp_size // pc.ffn_sp_size
        pc.ffn_tp_rank = pc.tp_rank % pc.ffn_tp_size if pc.ffn_tp_size else 0
        pc.enable_sp = pc.ffn_sp_size > 1
        return pc

    @timer_wrapper("convert 1 tp")
    def _convert(
        self,
        tp_rank: int,
        dp_rank: int,
        world_rank: int,
        output_dir_base: str,
        model_type: Optional[str] = None,
    ):
        model_type = model_type or self.model_type
        model_cls = ModelFactory.get_model_cls(model_type)
        is_draft = model_type != self.model_type
        ckpt_path = (
            self.draft_model_path
            if is_draft and self.draft_model_path
            else self.model_path
        )
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

        load_method = LoadMethod(
            str(env_params.get("LOAD_METHOD", LoadMethod.AUTO.value)).lower()
        )

        # Create config using _create_config
        model_config: ModelConfig = model_cls._create_config(ckpt_path)

        # Create ModelArgs from config
        model_args = ModelArgs()
        model_args.ckpt_path = ckpt_path
        model_args.tokenizer_path = ckpt_path
        model_args.model_type = model_type

        # A draft model is configured by the SP_* variables at serving time, and the
        # dump has to be produced under those same settings to be loadable.
        def draft_aware(key: str, default: str = "") -> str:
            if is_draft and f"SP_{key}" in env_params:
                return str(env_params[f"SP_{key}"])
            return str(env_params.get(key, default))

        model_args.act_type = draft_aware("ACT_TYPE")

        kv_cache_config = KVCacheConfig()
        kv_cache_config.seq_size_per_block = 64
        kv_cache_config.fp8_kv_cache = int(draft_aware("FP8_KV_CACHE", "0"))
        kv_cache_config.int8_kv_cache = int(draft_aware("INT8_KV_CACHE", "0"))

        quantization_config = QuantizationConfig()
        quantization_config.quantization = quantization

        # Build model config
        build_model_config(
            model_config=model_config,
            model_args=model_args,
            kv_cache_config=kv_cache_config,
            quantization_config=quantization_config,
            profiling_debug_logging_config=ProfilingDebugLoggingConfig(),
            embedding_config=None,  # Fake loader doesn't need embedding_config
        )

        # HACK_LAYER_NUM truncates the target's layer stack; a draft model has a
        # fixed layer count of its own that must not be overridden.
        if not is_draft:
            model_config.num_layers = int(
                env_params.get("HACK_LAYER_NUM", str(model_config.num_layers))
            )
        parallelism_config = self._build_parallelism_config()

        # Create other required configs
        hw_kernel_config = HWKernelConfig()
        fmha_config = FMHAConfig()
        moe_config = MoeConfig()
        device_resource_config = DeviceResourceConfig()
        vit_config = VitConfig()

        model = model_cls.from_config(
            model_config=model_config,
            parallelism_config=parallelism_config,
            hw_kernel_config=hw_kernel_config,
            kv_cache_config=kv_cache_config,
            fmha_config=fmha_config,
            moe_config=moe_config,
            load_method=load_method,
            max_generate_batch_size=0,
            vit_config=vit_config,
            merge_lora=False,
            device_resource_config=device_resource_config,
            force_cpu_load_weights=str(
                env_params.get("FORCE_CPU_LOAD_WEIGHTS", "0")
            ).lower()
            in ("1", "true"),
            skip_python_model=True,
        )
        loader = model.create_model_loader()
        device_str = f"cuda:{parallelism_config.local_rank}"
        max_retry_times = 3
        for i in range(max_retry_times):
            try:
                loader.dump_weight_as_ft_style(device_str, output_dir_base)
                logging.info(f"dump {model_type} rank:[{world_rank}] done")
                break
            except Exception as e:
                logging.warn(
                    f"dump {model_type} rank:[{world_rank}] failed, {str(e)}, retry {i} times"
                )
                if i == max_retry_times - 1:
                    logging.error(
                        f"dump {model_type} rank:[{world_rank}] retry {i} times, but still failed"
                    )
                    raise RuntimeError(f"Failed after 10 retries: {str(e)}") from e
                continue
        logging.info(f"convert model {model_type} rank:{world_rank} done")

    @timer_wrapper("save convert result")
    def _save_converted(self, input_path, output_path: str):
        self._copy_filtered_files(
            input_path,
            output_path,
            [
                "pytorch_model.bin.index.json",
                "model.safetensors.index.json",
                "*.safetensors",
                "*.bin",
                "*.pth",
                "*.pt",
                "*.gguf",
            ],
        )
        self._generate_safetensor_meta_info(output_path)
        # touch done
        done_file = os.path.join(output_path, "done")
        with open(done_file, "w") as f:
            pass

    @timer_wrapper("generate safetensors meta info")
    def _generate_safetensor_meta_info(self, output_path: str):
        weight_map = {}
        total_size = 0

        # 获取所有 .safetensor 文件，并按名称排序
        safetensor_files = sorted(
            [f for f in os.listdir(output_path) if f.endswith(".safetensors")]
        )

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
                            logging.warning(
                                f"警告: 权重 '{key}' 已经映射到 '{weight_map[key]}', 现在尝试映射到 '{st_file}'。"
                            )
                        weight_map[key] = st_file
            except Exception as e:
                logging.warning(f"无法读取文件 '{st_file}': {e}")

        pc = self._build_parallelism_config()
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
            "is_ft_style_weight": True,
            "ft_weight_sharding": {
                "attn_tp_size": int(pc.get_attn_tp_size()),
                "ffn_tp_size": int(pc.get_ffn_tp_size()),
                "dp_size": int(pc.dp_size),
                "ep_size": int(pc.ep_size),
                "lm_head_tp_size": int(pc.tp_size),
            },
            "ft_model_types": [
                t for t in (self.model_type, self.draft_model_type) if t
            ],
            "__env__params__": self.env_params,
        }

        # 将索引数据写入 JSON 文件
        output_file = "model.safetensors.index.json"
        output_path = os.path.join(output_path, output_file)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
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
    parser.add_argument(
        "--draft_model_type",
        type=str,
        default="",
        help="[Optional] speculative draft model to dump alongside the target; "
        "defaults to SP_MODEL_TYPE from --env_params.",
    )

    parser.add_argument(
        "--draft_pretrained_model_dir",
        type=str,
        default="",
        help="[Optional] HF checkpoint for the draft model when it differs from the "
        "target (e.g. MXFP8 target without bundled MTP weights).",
    )

    # 解析参数
    args = parser.parse_args()

    sys.argv = sys.argv[:1]

    env_params = json.loads(args.env_params)
    converter = WeightConverter(
        args.pretrained_model_dir,
        args.model_type,
        env_params,
        args.draft_model_type,
        args.draft_pretrained_model_dir or env_params.get("DRAFT_SRC"),
    )

    ret_code = converter.convert(args.output_dir_base)
    exit(ret_code)


if __name__ == "__main__":
    # logging.config.dictConfig(LOGGING_CONFIG)
    main()
