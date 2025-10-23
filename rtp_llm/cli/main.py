from rtp_llm.server.server_args.server_args import EnvArgumentParser

LLM_SUBCMD_PARSER_EPILOG = (
    "Tip: Use `rtp-llm [serve|run-batch|bench <bench_type>] "
    "--help=<keyword>` to explore arguments from help.\n"
    "   - To view a argument group:     --help=ModelConfig\n"
    "   - To view a single argument:    --help=max-num-seqs\n"
    "   - To search by keyword:         --help=max\n"
    "   - To list all groups:           --help=listgroup\n"
    "   - To view help with pager:      --help=page"
)

def main():
    import rtp_llm.cli.serve
    from rtp_llm.release_version import RELEASE_VERSION

    CMD_MODULES = [rtp_llm.cli.serve]

    parser = EnvArgumentParser(
        description="rtp-llm CLI",
        epilog=LLM_SUBCMD_PARSER_EPILOG,
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
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
