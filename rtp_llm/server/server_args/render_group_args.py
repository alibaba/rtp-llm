def init_render_group_args(parser):
    ##############################################################################################################
    # Render Configuration
    ##############################################################################################################
    render_group = parser.add_argument_group("Render Configuration")
    render_group.add_argument(
        "--model_template_type",
        env_name="MODEL_TEMPLATE_TYPE",
        type=str,
        default=None,
        help="模型的模版类型",
    )
    render_group.add_argument(
        "--default_chat_template_key",
        env_name="DEFAULT_CHAT_TEMPLATE_KEY",
        type=str,
        default="default",
        help="OpenAI的chat模型键",
    )
    render_group.add_argument(
        "--default_tool_use_template_key",
        env_name="DEFAULT_TOOL_USE_TEMPLATE_KEY",
        type=str,
        default="tool_use",
        help="默认工具使用的模板的key",
    )
    render_group.add_argument(
        "--llava_chat_template",
        env_name="LLAVA_CHAT_TEMPLATE",
        type=str,
        default="",
        help="LLava模型的会话模板",
    )
