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
            "确定性推理开关（默认关闭）。开启后强制单请求串行服务：每个请求的 prefill 单独执行、"
            "decode 以 1 real + (B_det-1) dummy 的固定组成运行，使任意凑批/并发组合下的输出与 solo "
            "bitwise 一致。性能显著下降（decode 不再跨请求合批），仅用于评测对拍等确定性场景。"
        ),
    )

    deterministic_group.add_argument(
        "--deterministic_level",
        env_name="DETERMINISTIC_LEVEL",
        bind_to=(deterministic_config, "level"),
        type=str,
        choices=["decode", "full"],
        default="full",
        help=(
            "确定性等级。full（默认）= 固定 decode 几何 + 单请求串行服务（prefill 独占 + "
            "max_generate_batch_size=1，输出与 solo bitwise 一致）；decode = 仅固定 decode 几何"
            "（单尺寸 CUDA Graph + B_det padding，消除几何类漂移，batch 组成相关的残余漂移仍在）。"
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
