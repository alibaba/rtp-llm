from rtp_llm.cli.utils import RTP_LLM_SUBCMD_PARSER_EPILOG
from rtp_llm.server.server_args.server_args import EnvArgumentParser



def main():
    import rtp_llm.cli.serve
    from rtp_llm.release_version import RELEASE_VERSION

    CMD_MODULES = [rtp_llm.cli.serve]

    parser = EnvArgumentParser(
        description="rtp-llm CLI",
        epilog=RTP_LLM_SUBCMD_PARSER_EPILOG,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=RELEASE_VERSION,
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(parser)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()