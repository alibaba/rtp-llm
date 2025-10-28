
import argparse
import typing

from rtp_llm.server.server_args.server_args import EnvArgumentParser

class CLISubcommand:
    """Base class for CLI argument handlers."""

    name: str

    @staticmethod
    def cmd(parser: EnvArgumentParser) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def validate(self, args: argparse.Namespace) -> None:
        # No validation by default
        pass

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> EnvArgumentParser:
        raise NotImplementedError("Subclasses should implement this method")
