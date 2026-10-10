import argparse
import json

from rtp_llm.config.module_dispatch_config import ModuleDispatchConfig


def parse_module_dispatch(value):
    try:
        return ModuleDispatchConfig.from_dict(json.loads(value))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def init_module_dispatch_group_args(parser, py_env_configs):
    group = parser.add_argument_group("Module construction")
    group.add_argument(
        "--module_dispatch",
        env_name="MODULE_DISPATCH",
        bind_to=(py_env_configs, "module_dispatch"),
        type=parse_module_dispatch,
        default=ModuleDispatchConfig(),
        help="JSON object: mode=legacy|auto, platform, impl_overrides, path_overrides",
    )
