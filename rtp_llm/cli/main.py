# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
'''The CLI entrypoints of vLLM

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage.'''
from __future__ import annotations


def main():
    import rtp_llm.cli.serve
    from utils import LLM_SUBCMD_PARSER_EPILOG, cli_env_setup
    from utils import FlexibleArgumentParser
    from rtp_llm.release_version import RELEASE_VERSION

    CMD_MODULES = [
        rtp_llm.cli.serve
    ]

    cli_env_setup()

    parser = FlexibleArgumentParser(
        description="rtp-llm CLI",
        epilog=LLM_SUBCMD_PARSER_EPILOG,
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=RELEASE_VERSION,
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(
                dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
