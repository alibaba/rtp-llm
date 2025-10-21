# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import signal
from typing import Optional
import logging


from rtp_llm.cli.types import CLISubcommand
from rtp_llm.cli.utils import FlexibleArgumentParser




class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI. """
    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        logging.info("call serve cmd")

    def validate(self, args: argparse.Namespace) -> None:
        logging.info("call serve validate")

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            "serve",
            help="Start the vLLM OpenAI Compatible API server.",
            description="Start the vLLM OpenAI Compatible API server.",
            usage="vllm serve [model_tag] [options]")

        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]


