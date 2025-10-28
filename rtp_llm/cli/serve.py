import argparse
import logging
import os

from rtp_llm.models.base_model import BaseModel
from rtp_llm.cli.types import CLISubcommand
from rtp_llm.cli.utils import show_filtered_argument_or_group_from_help
from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.server.server_args.server_args import EnvArgumentParser, init_all_group_args
from rtp_llm.start_server import start_server
from rtp_llm.tools.api.hf_model_helper import get_model_info_from_hf, get_hf_model_info





class ServeSubcommand(CLISubcommand):
    name = "serve"
    serve_sub_parser: EnvArgumentParser = None

    @staticmethod
    def cmd(parser: EnvArgumentParser) -> None:
        logging.info("start rtp serve cmd")
        args = parser.parse_args()

        if hasattr(args, 'model_tag') and args.model_tag is not None:
            model_path = args.model_tag
            setattr(args, "checkpoint_path", model_path)
            ServeSubcommand.serve_sub_parser.update_env_from_args("checkpoint_path", args)
        start_server()

    def validate(self, args: argparse.Namespace) -> None:
        logging.info("call serve validate")

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction, RTP_LLM_SUBCMD_PARSER_EPILOG=None) -> EnvArgumentParser:
        serve_parser = subparsers.add_parser(
            "serve",
            help="Start the RTP-LLM OpenAI Compatible API server.",
            description="Start the RTP-LLM OpenAI Compatible API server.",
            usage="rtp-llm serve [model_tag] [options]")
        serve_parser.add_argument("model_tag",
                                  type=str,
                                  nargs="?",
                                  help="The model tag to serve (optional if specified in config)")
        init_all_group_args(serve_parser)
        show_filtered_argument_or_group_from_help(serve_parser, ["serve"])
        serve_parser.epilog = RTP_LLM_SUBCMD_PARSER_EPILOG
        ServeSubcommand.serve_sub_parser = serve_parser
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]
