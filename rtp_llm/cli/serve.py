
import argparse
import logging

from rtp_llm.cli.types import CLISubcommand
from rtp_llm.server.server_args.server_args import EnvArgumentParser, init_all_group_args
from rtp_llm.start_server import start_server


class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI. """
    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        logging.info("start rtp serve cmd")
        start_server()

    def validate(self, args: argparse.Namespace) -> None:
        logging.info("call serve validate")

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> EnvArgumentParser:
        serve_parser = subparsers.add_parser(
            "serve",
            help="Start the RTP-LLM OpenAI Compatible API server.",
            description="Start the RTP-LLM OpenAI Compatible API server.",
            usage="rtp-llm serve [options]")
        init_all_group_args(serve_parser)

        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]