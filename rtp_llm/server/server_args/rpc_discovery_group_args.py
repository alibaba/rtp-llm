from rtp_llm.server.server_args.util import str2bool


def init_rpc_discovery_group_args(parser):
    ##############################################################################################################
    # RPC 与服务发现配置
    ##############################################################################################################
    rpc_discovery_group = parser.add_argument_group("RPC 与服务发现")

    rpc_discovery_group.add_argument(
        "--use_local",
        env_name="USE_LOCAL",
        type=str2bool,
        default=False,
        help="设置为 `True` 时，系统将使用本地配置进行 decode 和 ViT 服务发现，而不是依赖外部服务注册与发现机制 (如 CM2)。适用于本地测试或特定部署。",
    )

    rpc_discovery_group.add_argument(
        "--remote_rpc_server_ip",
        env_name="REMOTE_RPC_SERVER_IP",
        type=str,
        default=None,
        help="指定远程 RPC 服务器的 IP 地址和可选端口 (格式: `ip:port` 或 `ip`)。主要用于 prefill server 的本地测试和调试。",
    )

    rpc_discovery_group.add_argument(
        "--decode_cm2_config",
        env_name="RTP_LLM_DECODE_CM2_CONFIG",
        type=str,
        default=None,
        help="为 decode cluster 提供服务发现 (如 CM2) 的 JSON 配置字符串。用于在集群环境中定位 decode 服务。",
    )

    rpc_discovery_group.add_argument(
        "--remote_vit_server_ip",
        env_name="REMOTE_VIT_SERVER_IP",
        type=str,
        default=None,
        help="指定远程 ViT (Visual Transformer) 服务器的 IP 地址和可选端口。主要用于多模态模型的本地测试和调试。",
    )

    rpc_discovery_group.add_argument(
        "--multimodal_part_cm2_config",
        env_name="RTP_LLM_MULTIMODAL_PART_CM2_CONFIG",
        type=str,
        default=None,
        help="为多模态 (ViT) 服务部分提供服务发现 (如 CM2) 的 JSON 配置字符串。用于在集群环境中定位多模态处理服务。",
    )
