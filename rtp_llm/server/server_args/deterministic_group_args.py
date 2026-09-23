import logging

from rtp_llm.server.server_args.util import str2bool


def init_deterministic_group_args(parser, deterministic_config):
    ##############################################################################################################
    # 确定性推理开关
    ##############################################################################################################
    deterministic_group = parser.add_argument_group("Deterministic Inference")

    deterministic_group.add_argument(
        "--deterministic_inference",
        env_name="DETERMINISTIC_INFERENCE",
        bind_to=(deterministic_config, "enable"),
        type=str2bool,
        default=False,
        help=(
            "确定性推理开关（默认关闭）。开启后按 --deterministic_level 分级：固定 decode 几何"
            "（单尺寸 CUDA Graph + B_det padding），并按等级施加 prefill 独占/串行服务，使输出"
            "与 batch 组成无关（token 级一致；上报 logprob 数值存在 ~1e-6 组成噪声）。batched 档"
            "保留 decode 合批吞吐；full 档吞吐与并发解耦，仅用于评测对拍等场景。"
        ),
    )

    deterministic_group.add_argument(
        "--deterministic_level",
        env_name="DETERMINISTIC_LEVEL",
        bind_to=(deterministic_config, "level"),
        type=str,
        choices=["decode", "batched", "full"],
        default="full",
        help=(
            "确定性等级。batched = 生产确定性吞吐档：固定 decode 几何 + prefill 每 forward 单请求 + "
            "ratio 调度器 + decode 合批（B_det 封顶，超出排队），token 与 solo 一致；"
            "full（默认）= 固定 decode 几何 + 单请求串行服务（prefill 独占 + "
            "max_generate_batch_size=1，输出与 solo bitwise 一致）；decode = 仅固定 decode 几何"
            "（单尺寸 CUDA Graph + B_det padding，消除几何类漂移，prefill 合批相关的漂移仍在）。"
        ),
    )

    deterministic_group.add_argument(
        "--deterministic_decode_batch_size",
        env_name="DETERMINISTIC_DECODE_BATCH_SIZE",
        bind_to=(deterministic_config, "decode_batch_size"),
        type=int,
        default=8,
        help=(
            "确定性 decode 图的单尺寸捕获 batch size（B_det，默认 8）。decode batch 一律 pad 到 "
            "B_det（dummy 行 KV 长度约 1）。仅当 --deterministic_inference 开启时生效。"
        ),
    )

    logging.debug(
        "deterministic args registered: enable=%s level=%s decode_batch_size=%s",
        deterministic_config.enable,
        deterministic_config.level,
        deterministic_config.decode_batch_size,
    )
