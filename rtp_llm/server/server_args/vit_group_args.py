import os


def init_vit_group_args(parser):
    ##############################################################################################################
    # Vit Configuration
    ##############################################################################################################
    vit_group = parser.add_argument_group("Vit Configuration")
    vit_group.add_argument(
        "--vit_separation",
        env_name="VIT_SEPARATION",
        type=int,
        default=0,
        help="VIT是否和主进程进行分离",
    )
    vit_group.add_argument(
        "--vit_trt", env_name="VIT_TRT", type=int, default=0, help="VIT是否使用TRT库"
    )
    vit_group.add_argument(
        "--trt_cache_enabled",
        env_name="TRT_CACHE_ENABLED",
        type=int,
        default=0,
        help="是否使用TRT_CACHE",
    )
    vit_group.add_argument(
        "--trt_cache_path",
        env_name="TRT_CACHE_PATH",
        type=str,
        default=os.path.join(os.getcwd(), "trt_cache"),
        help="TRT_CACHE路径",
    )
    vit_group.add_argument(
        "--download_headers",
        env_name="DOWNLOAD_HEADERS",
        type=str,
        default="",
        help="是否需要下载headers",
    )
    vit_group.add_argument(
        "--mm_cache_item_num",
        env_name="MM_CACHE_ITEM_NUM",
        type=int,
        default=10,
        help="多模态开启的Cache的大小",
    )
    vit_group.add_argument(
        "--url_cache_item_num",
        env_name="URL_CACHE_ITEM_NUM",
        type=int,
        default=100,
        help="多模态开启的用于URL的Cache的大小",
    )
    vit_group.add_argument(
        "--use_igraph_cache",
        env_name="USE_IGRAPH_CACHE",
        type=bool,
        default=True,
        help="访问igraph是否开启cache",
    )
    vit_group.add_argument(
        "--igraph_search_dom",
        env_name="IGRAPH_SEARCH_DOM",
        type=str,
        default="com.taobao.search.igraph.common",
        help="访问igraph使用的vipserver地址",
    )
    vit_group.add_argument(
        "--igraph_vipserver",
        env_name="IGRAPH_VIPSERVER",
        type=int,
        default=0,
        help="是否使用vipserver访问igraph",
    )
    vit_group.add_argument(
        "--igraph_table_name",
        env_name="IGRAPH_TABLE_NAME",
        type=str,
        default="",
        help="igraph的表名",
    )
    vit_group.add_argument(
        "--igraph_default_key",
        env_name="IGRAPH_DEFAULT_KEY",
        type=str,
        default=None,
        help="访问igraph失败时默认使用的key",
    )
