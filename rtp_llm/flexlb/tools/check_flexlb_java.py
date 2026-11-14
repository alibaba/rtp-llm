"""Check member order and documentation for changed FlexLB Java classes.

pre-commit hides unstaged changes before supplying filenames. Documentation is
checked on classes intersecting the diff against HEAD, including their Javadoc.
Member order is checked on every supplied file. Accuracy and brevity of a class
description remain review responsibilities, not parser guarantees.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import order_java_members as members


def class_documentation(node, source):
    previous = node.prev_named_sibling
    if (
        previous is not None
        and previous.type == "block_comment"
        and source[previous.start_byte : previous.start_byte + 3] == b"/**"
        and not source[previous.end_byte : node.start_byte].strip()
    ):
        return previous
    return None


def is_data_only(node):
    """Recognize structural carriers; semantic constraints still need review."""
    body = node.child_by_field_name("body")
    if body is None:
        return False
    children = [
        child
        for child in body.named_children
        if child.type not in members.COMMENT_TYPES
    ]
    if node.type == "record_declaration":
        return not children
    if node.type == "enum_declaration":
        return all(
            child.type == "enum_constant"
            and all(
                part.type not in {"argument_list", "class_body"}
                for part in child.named_children
            )
            or child.type == "enum_body_declarations"
            and not child.named_children
            for child in children
        )
    if node.type != "class_declaration" or any(
        child.type in {"superclass", "super_interfaces"}
        for child in node.named_children
    ):
        return False
    return all(
        child.type == "field_declaration"
        and all(
            variable.child_by_field_name("value") is None
            for variable in child.named_children
            if variable.type == "variable_declarator"
        )
        for child in children
    )


def documentation_errors(source, changed_ranges):
    tree = members.PARSER.parse(source)
    errors = []

    def visit(node):
        if node.type in members.NESTED_TYPES:
            doc = class_documentation(node, source)
            first = (doc or node).start_point.row + 1
            last = node.end_point.row + 1
            affected = changed_ranges is None or any(
                start <= last and end >= first for start, end in changed_ranges
            )
            if affected:
                name_node = node.child_by_field_name("name")
                name = source[name_node.start_byte : name_node.end_byte].decode()
                message = None
                if doc is None and not is_data_only(node):
                    message = "add a brief class Javadoc describing its purpose or flow"
                elif doc is not None:
                    text = source[doc.start_byte : doc.end_byte].decode()
                    lines = [
                        re.sub(r"^\s*\* ?", "", line)
                        for line in text[3:-2].splitlines()
                    ]
                    summary = [re.sub(r"<[^>]*>", "", line).strip() for line in lines]
                    summary = [line for line in summary if line]
                    if not summary or summary[0].startswith("@"):
                        message = "class Javadoc needs a purpose or flow summary, not tags alone"
                    elif (
                        text.splitlines()[0].strip() != "/**"
                        or text.splitlines()[-1].strip() != "*/"
                    ):
                        message = "use multiline class Javadoc with /** and */ on separate lines"
                if message:
                    errors.append(f"{node.start_point.row + 1}: {name}: {message}")
        for child in node.named_children:
            visit(child)

    visit(tree.root_node)
    return errors


def changed_lines(path, repository):
    relative = path.resolve().relative_to(repository).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repository,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if tracked.returncode:
        return None
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--unified=0", "HEAD", "--", relative],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    ranges = []
    for match in re.finditer(
        r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.MULTILINE
    ):
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        # A pure deletion still changes the enclosing class at its surviving boundary.
        ranges.append((max(1, start), max(1, start + max(1, count) - 1)))
    return ranges


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    repository = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ).resolve()
    failed = False
    for path in args.paths:
        try:
            path.resolve().relative_to(repository / "rtp_llm/flexlb")
            source = path.read_bytes()
            if members.reorder(source) != source:
                print(
                    f"{path}: member order differs; run tools/order_java_members.py --apply on this file"
                )
                failed = True
            for error in documentation_errors(source, changed_lines(path, repository)):
                print(f"{path}:{error}")
                failed = True
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            print(f"{path}: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
