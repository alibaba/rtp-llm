from rtp_llm.server.server_args.util import str2bool


def init_fifo_scheduler_group_args(parser, fifo_scheduler_config):
    ##############################################################################################################
    # FIFO 调度器配置
    ##############################################################################################################
    fifo_scheduler_group = parser.add_argument_group("FIFO Scheduler")

    fifo_scheduler_group.add_argument(
        "--max_context_batch_size",
        env_name="MAX_CONTEXT_BATCH_SIZE",
        bind_to=[(fifo_scheduler_config, "max_context_batch_size")],
        type=int,
        default=1,
        help="（设备参数）为设备参数设置的最大 context batch size，影响默认调度器的凑批决策。",
    )
    fifo_scheduler_group.add_argument(
        "--max_batch_tokens_size",
        env_name="MAX_BATCH_TOKENS_SIZE",
        bind_to=[(fifo_scheduler_config, "max_batch_tokens_size")],
        type=int,
        default=0,
        help="最大 batch tokens 大小。",
    )
    fifo_scheduler_group.add_argument(
        "--pdfusion_scheduler_mode",
        env_name="PDFUSION_SCHEDULER_MODE",
        bind_to=[(fifo_scheduler_config, "pdfusion_scheduler_mode")],
        type=str,
        choices=["", "ratio"],
        default="",
        help="PDFUSION 调度器选择。默认空字符串使用 decode-first FIFO 调度；显式设置为 'ratio' 时启用 "
        "PDFusionRatioScheduler，并使用 decode_prefill_ratio 控制 prefill/decode 轮次比例。",
    )
    fifo_scheduler_group.add_argument(
        "--decode_prefill_ratio",
        env_name="DECODE_PREFILL_RATIO",
        bind_to=[(fifo_scheduler_config, "decode_prefill_ratio")],
        type=str,
        default="1",
        help="PDFusionRatioScheduler 的 decode:prefill 轮数比字符串。仅当 pdfusion_scheduler_mode='ratio' 生效；"
        "默认 '1' 严格交替；'N' = 1 轮 prefill 后 N 轮 decode；"
        "'1/X' = X 轮 prefill 后 1 轮 decode。非法值回退为 '1'。",
    )
    fifo_scheduler_group.add_argument(
        "--prefill_chunk_size",
        env_name="PREFILL_CHUNK_SIZE",
        bind_to=[(fifo_scheduler_config, "prefill_chunk_size")],
        type=int,
        default=0,
        help="chunked prefill 每个 chunk 的 token 数，>0 启用（自动对齐到 block）。"
        "仅 PREFILL / PDFUSION 角色，与 MLA / 线性注意力模型互斥。"
        "请求带 beam / logits / loss / hidden_states / all_probs / 多模态会被拒绝。",
    )
