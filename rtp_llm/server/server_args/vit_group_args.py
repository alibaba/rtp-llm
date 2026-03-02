import logging
import os

from rtp_llm.ops import VitSeparation


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
        default=0,
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
        type=bool,
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
        "--mm_batch_size",
        env_name="MM_BATCH_SIZE",
        type=int,
        default=1,
        help="多模态处理时批量大小",
    )
    vit_group.add_argument(
        "--biencoder_preprocess",
        env_name="BIENCODER_PREPROCESS",
        type=bool,
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
        type=bool,
        default=False,
        help="是否禁用访问日志",
    )
    vit_group.add_argument(
        "--use_local_preprocess",
        env_name="USE_LOCAL_PREPROCESS",
        bind_to=(vit_config, "use_local_preprocess"),
        type=bool,
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
