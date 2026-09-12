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
    for name, default, description in (
        (
            "vit_batch_wait_ms",
            5,
            "Maximum wait to collect a ViT GPU batch (milliseconds)",
        ),
        (
            "vit_max_batch_images",
            8,
            "Maximum images per ViT GPU forward; 1 disables batching",
        ),
        ("vit_max_batch_patches", 32768, "Maximum patches per ViT GPU forward"),
        (
            "vit_max_concurrent_requests",
            32,
            "Maximum in-flight requests per ViT worker",
        ),
        (
            "vit_token_cache_item_num",
            10000,
            "Maximum images in the independent CPU token-ID cache; 0 disables it",
        ),
        (
            "vit_token_cache_time_window_ms",
            30 * 60 * 1000,
            "Token-ID cache idle expiration in milliseconds; 0 disables it",
        ),
    ):
        vit_group.add_argument(
            "--" + name,
            env_name=name.upper(),
            bind_to=(vit_config, name),
            type=int,
            default=default,
            help=description,
        )
    for name, description in (
        (
            "mm_transport_mode",
            "ViT embedding transport: grpc (default), auto, or rdma (requires provider)",
        ),
        (
            "mm_rdma_bind_ip",
            "Routable ViT RDMA address; empty uses the host bind address",
        ),
        ("mm_rdma_port", "ViT RDMA listen port; 0 selects a port"),
        ("mm_rdma_connect_timeout_ms", "RDMA connection timeout in milliseconds"),
        (
            "mm_rdma_read_timeout_ms",
            "RDMA READ timeout, also capped by the RPC deadline",
        ),
        ("mm_rdma_release_timeout_ms", "Best-effort slot release RPC timeout"),
        (
            "mm_rdma_max_inflight_bytes",
            "Hard per-process registered GPU or pinned CPU pool cap",
        ),
        ("mm_rdma_max_slot_bytes", "Maximum bytes per image RDMA slot"),
    ):
        default = getattr(vit_config, name)
        vit_group.add_argument(
            "--" + name,
            env_name=name.upper(),
            bind_to=(vit_config, name),
            type=type(default),
            default=default,
            help=description,
        )
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
