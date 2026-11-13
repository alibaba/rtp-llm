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
        help=(
            "最大 context 请求数，通常默认 1。adaptive cadence 不使用此参数限制请求数；"
            "该模式按实际 packed token、总并发及 KV 预算准入，并从总并发推导容量配置。"
        ),
    )
    fifo_scheduler_group.add_argument(
        "--max_batch_tokens_size",
        env_name="MAX_BATCH_TOKENS_SIZE",
        bind_to=[(fifo_scheduler_config, "max_batch_tokens_size")],
        type=int,
        default=0,
        help=(
            "最大 batch tokens 大小。adaptive cadence 累加 "
            "contextLength * currentBatchSize，包含已保留/加载中的 Prefill；"
            "保留原有严格小于边界和单请求特例。legacy ratio 仍使用最长长度估算。"
        ),
    )
    fifo_scheduler_group.add_argument(
        "--max_batch_tokens_without_cache",
        env_name="MAX_BATCH_TOKENS_WITHOUT_CACHE",
        bind_to=[(fifo_scheduler_config, "max_batch_tokens_without_cache")],
        type=int,
        default=0,
        help=(
            "单轮调度中不含 KV cache prefix 的 prefill token 软上限；CP 按 Zigzag padding 后计数，"
            "保留越界 stream、停止 admit 后续 stream，但仍处理其错误状态；<=0 表示不限制。"
        ),
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
        "'0' = 有 waiting stream 时优先尝试 prefill；默认 '1' 严格交替；'N' = 1 轮 prefill 后 N 轮 decode；"
        "'1/X' = X 轮 prefill 后 1 轮 decode。非法值回退为 '1'。",
    )
    fifo_scheduler_group.add_argument(
        "--pdfusion_schedule_trace",
        env_name="PDFUSION_SCHEDULE_TRACE",
        bind_to=[(fifo_scheduler_config, "pdfusion_schedule_trace")],
        type=str2bool,
        default=False,
        help="Optional scheduling diagnostics, disabled by default. Cadence does not require tracing.",
    )
    fifo_scheduler_group.add_argument(
        "--pdfusion_trace_run_id",
        env_name="PDFUSION_TRACE_RUN_ID",
        bind_to=[(fifo_scheduler_config, "pdfusion_trace_run_id")],
        type=str,
        default="unset",
        help="Optional trace label only; never selects a control group. Cadence defaults to its internal group ID.",
    )
    fifo_scheduler_group.add_argument(
        "--pdfusion_coord_mode",
        env_name="PDFUSION_COORD_MODE",
        bind_to=[(fifo_scheduler_config, "pdfusion_coord_mode")],
        choices=["off", "cadence"],
        default="off",
        help="Opt-in single-host TP1/DP2/EP2 or TP1/DP4/EP4 global cadence; unsupported modes are rejected.",
    )
    fifo_scheduler_group.add_argument(
        "--pdfusion_coord_timeout_ms",
        env_name="PDFUSION_COORD_TIMEOUT_MS",
        bind_to=[(fifo_scheduler_config, "pdfusion_coord_timeout_ms")],
        type=int,
        default=30000,
        help="Control exchange deadline. Failure terminates the group without local fallback.",
    )
    fifo_scheduler_group.add_argument(
        "--cp_force_single_prefill",
        env_name="CP_FORCE_SINGLE_PREFILL",
        bind_to=[(fifo_scheduler_config, "cp_force_single_prefill")],
        type=str2bool,
        default=True,
        help="CP prefill 开启时是否强制每轮只调度一个 prefill 请求。",
    )
    fifo_scheduler_group.add_argument(
        "--max_inited_kv_cache_streams",
        env_name="MAX_INITED_KV_CACHE_STREAMS",
        bind_to=[(fifo_scheduler_config, "max_inited_kv_cache_streams")],
        type=int,
        default=0,
        help="FIFO 中已初始化 KV cache block 的 stream 数上限，0 表示不限制。",
    )
    fifo_scheduler_group.add_argument(
        "--prefill_chunk_size",
        env_name="PREFILL_CHUNK_SIZE",
        bind_to=[(fifo_scheduler_config, "prefill_chunk_size")],
        type=int,
        default=0,
        help="chunked prefill 单次 PREFILL forward 的总 token 预算，由所有真实 context row 共享。"
        ">0 时启用，并自动按 KV cache block 大小对齐。"
        "仅支持 PREFILL / PDFUSION 角色，支持普通 attention、MLA 和 hybrid 的分块调度。"
        "多返回序列时，若预算小于 num_return_sequences×seq_size_per_block，"
        "且缓存复用后的剩余输入大于预算//num_return_sequences，请求将在调度时以 INVALID_PARAMS 拒绝。"
        "请增大 prefill_chunk_size 或减小 num_return_sequences；预算建议至少为最大返回序列数×seq_size_per_block。"
        "使用 force_batch、beam、logits、loss、hidden_states、all_probs 或多模态输入的请求将被拒绝。",
    )
