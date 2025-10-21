# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import functools
import os
import subprocess
from typing import Any, Optional, Union

from fastapi import Request


from argparse import (Action, ArgumentDefaultsHelpFormatter, ArgumentParser,
                      ArgumentTypeError, RawDescriptionHelpFormatter,
                      _ArgumentGroup)
import regex as re
import textwrap
import sys
from collections import defaultdict
import yaml
import logging
import json

LLM_SUBCMD_PARSER_EPILOG = (
    "Tip: Use `rtp-vllm [serve|run-batch|bench <bench_type>] "
    "--help=<keyword>` to explore arguments from help.\n"
    "   - To view a argument group:     --help=ModelConfig\n"
    "   - To view a single argument:    --help=max-num-seqs\n"
    "   - To search by keyword:         --help=max\n"
    "   - To list all groups:           --help=listgroup\n"
    "   - To view help with pager:      --help=page")


async def listen_for_disconnect(request: Request) -> None:
    """Returns if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            # If load tracking is enabled *and* the counter exists, decrement
            # it. Combines the previous nested checks into a single condition
            # to satisfy the linter rule.
            if (getattr(request.app.state, "enable_server_load_tracking",
                        False)
                    and hasattr(request.app.state, "server_load_metrics")):
                request.app.state.server_load_metrics -= 1
            break


def with_cancellation(handler_func):
    """Decorator that allows a route handler to be cancelled by client
    disconnections.

    This does _not_ use request.is_disconnected, which does not work with
    middleware. Instead this follows the pattern from
    starlette.StreamingResponse, which simultaneously awaits on two tasks- one
    to wait for an http disconnect message, and the other to do the work that we
    want done. When the first task finishes, the other is cancelled.

    A core assumption of this method is that the body of the request has already
    been read. This is a safe assumption to make for fastapi handlers that have
    already parsed the body of the request into a pydantic model for us.
    This decorator is unsafe to use elsewhere, as it will consume and throw away
    all incoming messages for the request while it looks for a disconnect
    message.

    In the case where a `StreamingResponse` is returned by the handler, this
    wrapper will stop listening for disconnects and instead the response object
    will start listening for disconnects.
    """

    # Functools.wraps is required for this wrapper to appear to fastapi as a
    # normal route handler, with the correct request type hinting.
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):

        # The request is either the second positional arg or `raw_request`
        request = args[1] if len(args) > 1 else kwargs["raw_request"]

        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))

        done, pending = await asyncio.wait([handler_task, cancellation_task],
                                           return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()

        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


def decrement_server_load(request: Request):
    request.app.state.server_load_metrics -= 1



def cli_env_setup():
    # The safest multiprocessing method is `spawn`, as the default `fork` method
    # is not compatible with some accelerators. The default method will be
    # changing in future versions of Python, so we should use it explicitly when
    # possible.
    #
    # We only set it here in the CLI entrypoint, because changing to `spawn`
    # could break some existing code using vLLM as a library. `spawn` will cause
    # unexpected behavior if the code is not protected by
    # `if __name__ == "__main__":`.
    #
    # References:
    # - https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    # - https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    # - https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
    # - https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders
    if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
        logging.debug("Setting VLLM_WORKER_MULTIPROC_METHOD to 'spawn'")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _validate_truncation_size(
    max_model_len: int,
    truncate_prompt_tokens: Optional[int],
    tokenization_kwargs: Optional[dict[str, Any]] = None,
) -> Optional[int]:

    if truncate_prompt_tokens is not None:
        if truncate_prompt_tokens <= -1:
            truncate_prompt_tokens = max_model_len

        if truncate_prompt_tokens > max_model_len:
            raise ValueError(
                f"truncate_prompt_tokens value ({truncate_prompt_tokens}) "
                f"is greater than max_model_len ({max_model_len})."
                f" Please, select a smaller truncation size.")

        if tokenization_kwargs is not None:
            tokenization_kwargs["truncation"] = True
            tokenization_kwargs["max_length"] = truncate_prompt_tokens

    else:
        if tokenization_kwargs is not None:
            tokenization_kwargs["truncation"] = False

    return truncate_prompt_tokens


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


def show_filtered_argument_or_group_from_help(parser: argparse.ArgumentParser,
                                              subcommand_name: list[str]):

    # Only handle --help=<keyword> for the current subcommand.
    # Since subparser_init() runs for all subcommands during CLI setup,
    # we skip processing if the subcommand name is not in sys.argv.
    # sys.argv[0] is the program name. The subcommand follows.
    # e.g., for `vllm bench latency`,
    # sys.argv is `['vllm', 'bench', 'latency', ...]`
    # and subcommand_name is "bench latency".
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



class StoreBoolean(Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower() == "true":
            setattr(namespace, self.dest, True)
        elif values.lower() == "false":
            setattr(namespace, self.dest, False)
        else:
            raise ValueError(f"Invalid boolean value: {values}. "
                             "Expected 'true' or 'false'.")
class SortedHelpFormatter(ArgumentDefaultsHelpFormatter,
                          RawDescriptionHelpFormatter):
    """SortedHelpFormatter that sorts arguments by their option strings."""

    def _split_lines(self, text, width):
        """
        1. Sentences split across lines have their single newlines removed.
        2. Paragraphs and explicit newlines are split into separate lines.
        3. Each line is wrapped to the specified width (width of terminal).
        """
        # The patterns also include whitespace after the newline
        single_newline = re.compile(r"(?<!\n)\n(?!\n)\s*")
        multiple_newlines = re.compile(r"\n{2,}\s*")
        text = single_newline.sub(' ', text)
        lines = re.split(multiple_newlines, text)
        return sum([textwrap.wrap(line, width) for line in lines], [])

    def add_arguments(self, actions):
        actions = sorted(actions, key=lambda x: x.option_strings)
        super().add_arguments(actions)

class FlexibleArgumentParser(ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    _deprecated: set[Action] = set()
    _json_tip: str = (
        "When passing JSON CLI arguments, the following sets of arguments "
        "are equivalent:\n"
        '   --json-arg \'{"key1": "value1", "key2": {"key3": "value2"}}\'\n'
        "   --json-arg.key1 value1 --json-arg.key2.key3 value2\n\n"
        "Additionally, list elements can be passed individually using +:\n"
        '   --json-arg \'{"key4": ["value3", "value4", "value5"]}\'\n'
        "   --json-arg.key4+ value3 --json-arg.key4+=\'value4,value5\'\n\n")

    def __init__(self, *args, **kwargs):
        # Set the default "formatter_class" to SortedHelpFormatter
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = SortedHelpFormatter
        super().__init__(*args, **kwargs)

    if sys.version_info < (3, 13):
        # Enable the deprecated kwarg for Python 3.12 and below

        def parse_known_args(self, args=None, namespace=None):
            if args is not None and "--disable-log-requests" in args:
                # Special case warning because the warning below won't trigger
                # if –-disable-log-requests because its value is default.
                logging.warning(
                    "argument '--disable-log-requests' is deprecated and "
                    "replaced with '--enable-log-requests'. This will be "
                    "removed in v0.12.0.")
            namespace, args = super().parse_known_args(args, namespace)
            for action in FlexibleArgumentParser._deprecated:
                if (hasattr(namespace, dest := action.dest)
                        and getattr(namespace, dest) != action.default):
                    logging.warning("argument '%s' is deprecated", dest)
            return namespace, args

        def add_argument(self, *args, **kwargs):
            deprecated = kwargs.pop("deprecated", False)
            action = super().add_argument(*args, **kwargs)
            if deprecated:
                FlexibleArgumentParser._deprecated.add(action)
            return action

        class _FlexibleArgumentGroup(_ArgumentGroup):

            def add_argument(self, *args, **kwargs):
                deprecated = kwargs.pop("deprecated", False)
                action = super().add_argument(*args, **kwargs)
                if deprecated:
                    FlexibleArgumentParser._deprecated.add(action)
                return action

        def add_argument_group(self, *args, **kwargs):
            group = self._FlexibleArgumentGroup(self, *args, **kwargs)
            self._action_groups.append(group)
            return group

    def format_help(self) -> str:
        # Add tip about JSON arguments to the epilog
        epilog = self.epilog or ""
        if not epilog.startswith(FlexibleArgumentParser._json_tip):
            self.epilog = FlexibleArgumentParser._json_tip + epilog
        return super().format_help()

    def parse_args(  # type: ignore[override]
        self,
        args: list[str] | None = None,
        namespace: Namespace | None = None,
    ):
        if args is None:
            args = sys.argv[1:]

        # Check for --model in command line arguments first
        if args and args[0] == "serve":
            model_in_cli_args = any(arg == '--model' for arg in args)

            if model_in_cli_args:
                raise ValueError(
                    "With `vllm serve`, you should provide the model as a "
                    "positional argument or in a config file instead of via "
                    "the `--model` option.")

        if '--config' in args:
            args = self._pull_args_from_config(args)

        def repl(match: re.Match) -> str:
            """Replaces underscores with dashes in the matched string."""
            return match.group(0).replace("_", "-")

        # Everything between the first -- and the first .
        pattern = re.compile(r"(?<=--)[^\.]*")

        # Convert underscores to dashes and vice versa in argument names
        processed_args = list[str]()
        for i, arg in enumerate(args):
            if arg.startswith('--'):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = pattern.sub(repl, key, count=1)
                    processed_args.append(f'{key}={value}')
                else:
                    key = pattern.sub(repl, arg, count=1)
                    processed_args.append(key)
            elif arg.startswith('-O') and arg != '-O' and arg[2] != '.':
                # allow -O flag to be used without space, e.g. -O3 or -Odecode
                # -O.<...> handled later
                # also handle -O=<level> here
                level = arg[3:] if arg[2] == '=' else arg[2:]
                processed_args.append(f'-O.level={level}')
            elif arg == '-O' and i + 1 < len(args) and args[i + 1] in {
                    "0", "1", "2", "3"
            }:
                # Convert -O <n> to -O.level <n>
                processed_args.append('-O.level')
            else:
                processed_args.append(arg)

        def create_nested_dict(keys: list[str], value: str) -> dict[str, Any]:
            """Creates a nested dictionary from a list of keys and a value.

            For example, `keys = ["a", "b", "c"]` and `value = 1` will create:
            `{"a": {"b": {"c": 1}}}`
            """
            nested_dict: Any = value
            for key in reversed(keys):
                nested_dict = {key: nested_dict}
            return nested_dict

        def recursive_dict_update(
            original: dict[str, Any],
            update: dict[str, Any],
        ) -> set[str]:
            """Recursively updates a dictionary with another dictionary.
            Returns a set of duplicate keys that were overwritten.
            """
            duplicates = set[str]()
            for k, v in update.items():
                if isinstance(v, dict) and isinstance(original.get(k), dict):
                    nested_duplicates = recursive_dict_update(original[k], v)
                    duplicates |= {f"{k}.{d}" for d in nested_duplicates}
                elif isinstance(v, list) and isinstance(original.get(k), list):
                    original[k] += v
                else:
                    if k in original:
                        duplicates.add(k)
                    original[k] = v
            return duplicates

        delete = set[int]()
        dict_args = defaultdict[str, dict[str, Any]](dict)
        duplicates = set[str]()
        for i, processed_arg in enumerate(processed_args):
            if i in delete:  # skip if value from previous arg
                continue

            if processed_arg.startswith("-") and "." in processed_arg:
                if "=" in processed_arg:
                    processed_arg, value_str = processed_arg.split("=", 1)
                    if "." not in processed_arg:
                        # False positive, '.' was only in the value
                        continue
                else:
                    value_str = processed_args[i + 1]
                    delete.add(i + 1)

                if processed_arg.endswith("+"):
                    processed_arg = processed_arg[:-1]
                    value_str = json.dumps(list(value_str.split(",")))

                key, *keys = processed_arg.split(".")
                try:
                    value = json.loads(value_str)
                except json.decoder.JSONDecodeError:
                    value = value_str

                # Merge all values with the same key into a single dict
                arg_dict = create_nested_dict(keys, value)
                arg_duplicates = recursive_dict_update(dict_args[key],
                                                       arg_dict)
                duplicates |= {f'{key}.{d}' for d in arg_duplicates}
                delete.add(i)
        # Filter out the dict args we set to None
        processed_args = [
            a for i, a in enumerate(processed_args) if i not in delete
        ]
        if duplicates:
            logging.warning("Found duplicate keys %s", ", ".join(duplicates))

        # Add the dict args back as if they were originally passed as JSON
        for dict_arg, dict_value in dict_args.items():
            processed_args.append(dict_arg)
            processed_args.append(json.dumps(dict_value))

        return super().parse_args(processed_args, namespace)

    def check_port(self, value):
        try:
            value = int(value)
        except ValueError:
            msg = "Port must be an integer"
            raise ArgumentTypeError(msg) from None

        if not (1024 <= value <= 65535):
            raise ArgumentTypeError("Port must be between 1024 and 65535")

        return value

    def _pull_args_from_config(self, args: list[str]) -> list[str]:
        """Method to pull arguments specified in the config file
        into the command-line args variable.

        The arguments in config file will be inserted between
        the argument list.

        example:
        ```yaml
            port: 12323
            tensor-parallel-size: 4
        ```
        ```python
        $: vllm {serve,chat,complete} "facebook/opt-12B" \
            --config config.yaml -tp 2
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--config', 'config.yaml',
            '-tp', '2'
        ]
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--port', '12323',
            '--tensor-parallel-size', '4',
            '-tp', '2'
            ]
        ```

        Please note how the config args are inserted after the sub command.
        this way the order of priorities is maintained when these are args
        parsed by super().
        """
        assert args.count(
            '--config') <= 1, "More than one config file specified!"

        index = args.index('--config')
        if index == len(args) - 1:
            raise ValueError("No config file specified! \
                             Please check your command-line arguments.")

        file_path = args[index + 1]

        config_args = self._load_config_file(file_path)

        # 0th index is for {serve,chat,complete}
        # optionally followed by model_tag (only for serve)
        # followed by config args
        # followed by rest of cli args.
        # maintaining this order will enforce the precedence
        # of cli > config > defaults
        if args[0] == "serve":
            model_in_cli = len(args) > 1 and not args[1].startswith('-')
            model_in_config = any(arg == '--model' for arg in config_args)

            if not model_in_cli and not model_in_config:
                raise ValueError(
                    "No model specified! Please specify model either "
                    "as a positional argument or in a config file.")

            if model_in_cli:
                # Model specified as positional arg, keep CLI version
                args = [args[0]] + [
                    args[1]
                ] + config_args + args[2:index] + args[index + 2:]
            else:
                # No model in CLI, use config if available
                args = [args[0]
                        ] + config_args + args[1:index] + args[index + 2:]
        else:
            args = [args[0]] + config_args + args[1:index] + args[index + 2:]

        return args

    def _load_config_file(self, file_path: str) -> list[str]:
        """Loads a yaml file and returns the key value pairs as a
        flattened list with argparse like pattern
        ```yaml
            port: 12323
            tensor-parallel-size: 4
        ```
        returns:
            processed_args: list[str] = [
                '--port': '12323',
                '--tensor-parallel-size': '4'
            ]
        """
        extension: str = file_path.split('.')[-1]
        if extension not in ('yaml', 'yml'):
            raise ValueError(
                "Config file must be of a yaml/yml type.\
                              %s supplied", extension)

        # only expecting a flat dictionary of atomic types
        processed_args: list[str] = []

        config: dict[str, Union[int, str]] = {}
        try:
            with open(file_path) as config_file:
                config = yaml.safe_load(config_file)
        except Exception as ex:
            logging.error(
                "Unable to read the config file at %s. \
                Make sure path is correct", file_path)
            raise ex

        store_boolean_arguments = [
            action.dest for action in self._actions
            if isinstance(action, StoreBoolean)
        ]

        for key, value in config.items():
            if isinstance(value, bool) and key not in store_boolean_arguments:
                if value:
                    processed_args.append('--' + key)
            else:
                processed_args.append('--' + key)
                processed_args.append(str(value))

        return processed_args