
import argparse
import logging

from rtp_llm.cli.types import CLISubcommand
from rtp_llm.cli.utils import show_filtered_argument_or_group_from_help
from rtp_llm.server.server_args.server_args import EnvArgumentParser, init_all_group_args
from rtp_llm.start_server import start_server


class ServeSubcommand(CLISubcommand):
    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        logging.info("start rtp serve cmd")
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
            usage="rtp-llm serve [options]")
        init_all_group_args(serve_parser)
        show_filtered_argument_or_group_from_help(serve_parser, ["serve"])
        serve_parser.epilog = RTP_LLM_SUBCMD_PARSER_EPILOG

        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]