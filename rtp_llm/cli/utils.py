import argparse
import subprocess
import sys

RTP_LLM_SUBCMD_PARSER_EPILOG = (
    "Tip: Use `rtp-llm [serve] "
    "--help=<keyword>` to explore arguments from help.\n"
    "   - To view a argument group:     --help='Model Configuration'\n"
    "   - To view a single argument:    --help='batch_decode_scheduler_batch_size'\n"
    "   - To search by keyword:         --help=max\n"
    "   - To list all groups:           --help=listgroup\n"
    "   - To view help with pager:      --help=page"
)

def show_filtered_argument_or_group_from_help(parser: argparse.ArgumentParser,
                                              subcommand_name: list[str]):

    if len(sys.argv) <= len(subcommand_name) or sys.argv[
            1:1 + len(subcommand_name)] != subcommand_name:
        return

    for arg in sys.argv:
        if arg.startswith('--help='):
            search_keyword = arg.split('=', 1)[1]

            # Enable paged view for full help
            if search_keyword == 'page':
                help_text = parser.format_help()
                _output_with_pager(help_text)
                sys.exit(0)

            # List available groups
            if search_keyword == 'listgroup':
                output_lines = ["\nAvailable argument groups:"]
                for group in parser._action_groups:
                    if group.title and not group.title.startswith(
                            "positional arguments"):
                        output_lines.append(f"  - {group.title}")
                        if group.description:
                            output_lines.append("    " +
                                                group.description.strip())
                        output_lines.append("")
                _output_with_pager("\n".join(output_lines))
                sys.exit(0)

            # For group search
            formatter = parser._get_formatter()
            for group in parser._action_groups:
                if group.title and group.title.lower() == search_keyword.lower(
                ):
                    formatter.start_section(group.title)
                    formatter.add_text(group.description)
                    formatter.add_arguments(group._group_actions)
                    formatter.end_section()
                    _output_with_pager(formatter.format_help())
                    sys.exit(0)

            # For single arg
            matched_actions = []

            for group in parser._action_groups:
                for action in group._group_actions:
                    # search option name
                    if any(search_keyword.lower() in opt.lower()
                           for opt in action.option_strings):
                        matched_actions.append(action)

            if matched_actions:
                header = f"\nParameters matching '{search_keyword}':\n"
                formatter = parser._get_formatter()
                formatter.add_arguments(matched_actions)
                _output_with_pager(header + formatter.format_help())
                sys.exit(0)

            print(f"\nNo group or parameter matching '{search_keyword}'")
            print("Tip: use `--help=listgroup` to view all groups.")
            sys.exit(1)

def _output_with_pager(text: str):
    """Output text using scrolling view if available and appropriate."""

    pagers = ['less -R', 'more']
    for pager_cmd in pagers:
        try:
            proc = subprocess.Popen(pager_cmd.split(),
                                    stdin=subprocess.PIPE,
                                    text=True)
            proc.communicate(input=text)
            return
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            continue

    # No pager worked, fall back to normal print
    print(text)