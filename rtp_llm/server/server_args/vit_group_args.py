import logging
import os

from rtp_llm.ops import VitSeparation
from rtp_llm.server.server_args.util import str2bool


def _convert_vit_separation(value):
    """Convert value to VitSeparation enum.
    Accepts:
      - int: 0, 1, 2
      - str: "0", "1", "2"
      - str: "VitSeparation.VIT_SEPARATION_LOCAL"
             "VitSeparation.VIT_SEPARATION_ROLE"
             "VitSeparation.VIT_SEPARATION_REMOTE"
    """
    if isinstance(value, int):
        if value == 0:
            return VitSeparation.VIT_SEPARATION_LOCAL
        elif value == 1:
            return VitSeparation.VIT_SEPARATION_ROLE
        elif value == 2:
            return VitSeparation.VIT_SEPARATION_REMOTE

    if isinstance(value, str):
        value = value.strip()
        if value == "0" or value == "VitSeparation.VIT_SEPARATION_LOCAL":
            return VitSeparation.VIT_SEPARATION_LOCAL
        elif value == "1" or value == "VitSeparation.VIT_SEPARATION_ROLE":
            return VitSeparation.VIT_SEPARATION_ROLE
        elif value == "2" or value == "VitSeparation.VIT_SEPARATION_REMOTE":
            return VitSeparation.VIT_SEPARATION_REMOTE

    raise ValueError(
        f"Invalid vit_separation value: '{value}'. "
        f"Must be one of:\n"
        f"  0, 1, 2\n"
        f"  '0', '1', '2'\n"
        f"  'VitSeparation.VIT_SEPARATION_LOCAL'\n"
        f"  'VitSeparation.VIT_SEPARATION_ROLE'\n"
        f"  'VitSeparation.VIT_SEPARATION_REMOTE'"
    )


def init_vit_group_args(parser, vit_config):
    ##############################################################################################################
    # Vit Configuration
    ##############################################################################################################
    vit_group = parser.add_argument_group("Vit Configuration")
    vit_group.add_argument(
        "--vit_separation",
        env_name="VIT_SEPARATION",
        bind_to=(vit_config, "vit_separation"),
        type=_convert_vit_separation,
        default=VitSeparation.VIT_SEPARATION_LOCAL,
        help="VIT是否和主进程进行分离",
    )
    vit_group.add_argument(
        "--vit_trt",
        env_name="VIT_TRT",
        bind_to=(vit_config, "vit_trt"),
        type=int,
        default=0,
        help="VIT是否使用TRT库",
    )
    vit_group.add_argument(
        "--trt_cache_enabled",
        env_name="TRT_CACHE_ENABLED",
        bind_to=(vit_config, "trt_cache_enabled"),
        type=int,
        default=0,
        help="是否使用TRT_CACHE",
    )
    vit_group.add_argument(
        "--trt_cache_path",
        env_name="TRT_CACHE_PATH",
        bind_to=(vit_config, "trt_cache_path"),
        type=str,
        default=os.path.join(os.getcwd(), "trt_cache"),
        help="TRT_CACHE路径",
    )
    vit_group.add_argument(
        "--download_headers",
        env_name="DOWNLOAD_HEADERS",
        bind_to=(vit_config, "download_headers"),
        type=str,
        default="",
        help="是否需要下载headers",
    )
    vit_group.add_argument(
        "--mm_cache_item_num",
        env_name="MM_CACHE_ITEM_NUM",
        bind_to=(vit_config, "mm_cache_item_num"),
        type=int,
        default=10,
        help="多模态开启的Cache的大小",
    )
    vit_group.add_argument(
        "--url_cache_item_num",
        env_name="URL_CACHE_ITEM_NUM",
        bind_to=(vit_config, "url_cache_item_num"),
        type=int,
        default=100,
        help="多模态开启的用于URL的Cache的大小",
    )
    vit_group.add_argument(
        "--use_igraph_cache",
        env_name="USE_IGRAPH_CACHE",
        bind_to=(vit_config, "use_igraph_cache"),
        type=str2bool,
        default=True,
        help="访问igraph是否开启cache",
    )
    vit_group.add_argument(
        "--igraph_search_dom",
        env_name="IGRAPH_SEARCH_DOM",
        bind_to=(vit_config, "igraph_search_dom"),
        type=str,
        default="com.taobao.search.igraph.common",
        help="访问igraph使用的vipserver地址",
    )
    vit_group.add_argument(
        "--igraph_vipserver",
        env_name="IGRAPH_VIPSERVER",
        bind_to=(vit_config, "igraph_vipserver"),
        type=int,
        default=0,
        help="是否使用vipserver访问igraph",
    )
    vit_group.add_argument(
        "--igraph_table_name",
        env_name="IGRAPH_TABLE_NAME",
        bind_to=(vit_config, "igraph_table_name"),
        type=str,
        default="",
        help="igraph的表名",
    )
    vit_group.add_argument(
        "--igraph_default_key",
        env_name="IGRAPH_DEFAULT_KEY",
        bind_to=(vit_config, "default_key"),
        type=str,
        default=None,
        help="访问igraph失败时默认使用的key",
    )
    vit_group.add_argument(
        "--mm_preprocess_max_workers",
        env_name="MM_PREPROCESS_MAX_WORKERS",
        bind_to=(vit_config, "mm_preprocess_max_workers"),
        type=int,
        default=4,
        help="多模态预处理时最大线程数量",
    )
    vit_group.add_argument(
        "--biencoder_preprocess",
        env_name="BIENCODER_PREPROCESS",
        bind_to=(vit_config, "biencoder_preprocess"),
        type=str2bool,
        default=False,
        help="是否开启biencoder预处理",
    )
    vit_group.add_argument(
        "--extra_input_in_mm_embedding",
        env_name="EXTRA_INPUT_IN_MM_EMBEDDING",
        bind_to=(vit_config, "extra_input_in_mm_embedding"),
        type=str,
        default=None,
        help='在多模态嵌入中使用额外的输入，可选值"INDEX"',
    )
    vit_group.add_argument(
        "--mm_timeout_ms",
        env_name="MM_TIMEOUT_MS",
        bind_to=(vit_config, "mm_timeout_ms"),
        type=int,
        default=120000,
        help="多模态嵌入的超时时间，单位为毫秒",
    )
    vit_group.add_argument(
        "--extra_data_path",
        env_name="EXTRA_DATA_PATH",
        bind_to=(vit_config, "extra_data_path"),
        type=str,
        default="",
        help="额外的数据路径",
    )
    vit_group.add_argument(
        "--local_extra_data_path",
        env_name="LOCAL_EXTRA_DATA_PATH",
        bind_to=(vit_config, "local_extra_data_path"),
        type=str,
        default="",
        help="本地额外数据路径",
    )
    vit_group.add_argument(
        "--disable_access_log",
        env_name="DISABLE_ACCESS_LOG",
        bind_to=(vit_config, "disable_access_log"),
        type=str2bool,
        default=False,
        help="是否禁用访问日志",
    )
    vit_group.add_argument(
        "--use_local_preprocess",
        env_name="USE_LOCAL_PREPROCESS",
        bind_to=(vit_config, "use_local_preprocess"),
        type=str2bool,
        default=False,
        help="是否使用本地预处理模式（不使用子进程）",
    )
    vit_group.add_argument(
        "--vit_proxy_load_balance_strategy",
        env_name="VIT_PROXY_LOAD_BALANCE_STRATEGY",
        bind_to=(vit_config, "vit_proxy_load_balance_strategy"),
        type=str,
        default="round_robin",
        help="VIT代理服务器的负载均衡策略，可选值: 'round_robin' 或 'least_connections'",
    )
    # ---- Encoder(ViT)<->LLM embedding transport over GPUDirect RDMA ----
    vit_group.add_argument(
        "--mm_rdma_enable",
        env_name="MM_RDMA_ENABLE",
        bind_to=(vit_config, "mm_rdma_enable"),
        type=str2bool,
        default=False,
        help="开启 encoder<->LLM 多模态 embedding 的 GPUDirect RDMA 数据面（关闭时回退 bytes）",
    )
    vit_group.add_argument(
        "--mm_rdma_bind_ip",
        env_name="MM_RDMA_BIND_IP",
        bind_to=(vit_config, "mm_rdma_bind_ip"),
        type=str,
        default="",
        help="encoder 侧 RdmaServer 的 OOB 监听 IP，空表示自动探测（getBindIp）；多网卡机器可显式指定",
    )
    vit_group.add_argument(
        "--mm_rdma_port",
        env_name="MM_RDMA_PORT",
        bind_to=(vit_config, "mm_rdma_port"),
        type=int,
        default=0,
        help="encoder 侧 RdmaServer 监听端口，0 表示随机端口",
    )
    vit_group.add_argument(
        "--mm_rdma_min_bytes",
        env_name="MM_RDMA_MIN_BYTES",
        bind_to=(vit_config, "mm_rdma_min_bytes"),
        type=int,
        default=256 * 1024,
        help="只有大于该字节数的 embedding 才走 RDMA，更小的走 bytes",
    )
    vit_group.add_argument(
        "--mm_rdma_connect_timeout_ms",
        env_name="MM_RDMA_CONNECT_TIMEOUT_MS",
        bind_to=(vit_config, "mm_rdma_connect_timeout_ms"),
        type=int,
        default=250,
        help="RDMA 连接超时（毫秒）",
    )
    vit_group.add_argument(
        "--mm_rdma_read_timeout_ms",
        env_name="MM_RDMA_READ_TIMEOUT_MS",
        bind_to=(vit_config, "mm_rdma_read_timeout_ms"),
        type=int,
        default=30 * 1000,
        help="LLM 侧单次 RDMA READ 完成的最大等待时间（毫秒）",
    )
    vit_group.add_argument(
        "--mm_rdma_slot_gc_timeout_ms",
        env_name="MM_RDMA_SLOT_GC_TIMEOUT_MS",
        bind_to=(vit_config, "mm_rdma_slot_gc_timeout_ms"),
        type=int,
        default=60 * 1000,
        help="encoder 侧 slot 在无 Release 时强制回收的超时（毫秒）",
    )
    vit_group.add_argument(
        "--mm_rdma_max_inflight_bytes",
        env_name="MM_RDMA_MAX_INFLIGHT_BYTES",
        bind_to=(vit_config, "mm_rdma_max_inflight_bytes"),
        type=int,
        default=8 * 1024 * 1024 * 1024,
        help="encoder 侧在途（已注册未释放）embedding slot 的总字节软上限，超过则该次回退 bytes；0 表示不限制",
    )
