def init_render_group_args(parser, render_config):
    ##############################################################################################################
    # Render Configuration
    ##############################################################################################################
    render_group = parser.add_argument_group("Render Configuration")
    render_group.add_argument(
        "--model_template_type",
        env_name="MODEL_TEMPLATE_TYPE",
        bind_to=(render_config, 'model_template_type'),
        type=str,
        default=None,
        help="模型的模版类型",
    )
    render_group.add_argument(
        "--default_chat_template_key",
        env_name="DEFAULT_CHAT_TEMPLATE_KEY",
        bind_to=(render_config, 'default_chat_template_key'),
        type=str,
        default="default",
        help="OpenAI的chat模型键",
    )
    render_group.add_argument(
        "--default_tool_use_template_key",
        env_name="DEFAULT_TOOL_USE_TEMPLATE_KEY",
        bind_to=(render_config, 'default_tool_use_template_key'),
        type=str,
        default="tool_use",
        help="默认工具使用的模板的key",
    )
    render_group.add_argument(
        "--llava_chat_template",
        env_name="LLAVA_CHAT_TEMPLATE",
        bind_to=(render_config, 'llava_chat_template'),
        type=str,
        default="",
        help="LLava模型的会话模板",
    )
