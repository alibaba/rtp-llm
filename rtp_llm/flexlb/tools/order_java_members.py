#!/usr/bin/env python3
"""Order Java members without changing method contents or initializer order.

Default: check all FlexLB Java sources. Use --apply to write changes, then run Spotless.
Requires the pinned packages in java-member-order-requirements.txt.
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

try:
    import tree_sitter_java
    from tree_sitter import Language, Parser
except ImportError:
    raise SystemExit(
        "Install dependencies: python3 -m pip install -r "
        "rtp_llm/flexlb/tools/java-member-order-requirements.txt"
    )

PARSER = Parser(Language(tree_sitter_java.language()))

BODY_TYPES = {
    "class_body",
    "interface_body",
    "annotation_type_body",
    "enum_body_declarations",
}
METHOD_TYPES = {"method_declaration", "annotation_type_element_declaration"}
CONSTRUCTOR_TYPES = {"constructor_declaration", "compact_constructor_declaration"}
NESTED_TYPES = {
    "class_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
    "annotation_type_declaration",
}
COMMENT_TYPES = {"line_comment", "block_comment"}


def visibility(node, src):
    mod = next((c for c in node.children if c.type == "modifiers"), None)
    words = set(src[mod.start_byte : mod.end_byte].decode().split()) if mod else set()
    return (
        0
        if "public" in words
        else 1 if "protected" in words else 3 if "private" in words else 2
    )


def canonical(node, src):
    if node.type in COMMENT_TYPES:
        return [node.type, src[node.start_byte : node.end_byte].decode()]
    if not node.children:
        return [node.type, src[node.start_byte : node.end_byte].decode()]
    children = [canonical(c, src) for c in node.children]
    if node.type in BODY_TYPES:

        def static_member(c):
            if c.type == "static_initializer":
                return True
            mod = next((x for x in c.children if x.type == "modifiers"), None)
            return mod is not None and bool(
                re.search(rb"\bstatic\b", src[mod.start_byte : mod.end_byte])
            )

        fixed_nodes = [
            (c, v)
            for c, v in zip(node.children, children)
            if c.type
            not in METHOD_TYPES | CONSTRUCTOR_TYPES | NESTED_TYPES | COMMENT_TYPES
        ]
        fixed = [
            [v for c, v in fixed_nodes if static_member(c)],
            [v for c, v in fixed_nodes if not static_member(c)],
        ]
        moved = sorted(
            (
                v
                for c, v in zip(node.children, children)
                if c.type
                in METHOD_TYPES | CONSTRUCTOR_TYPES | NESTED_TYPES | COMMENT_TYPES
            ),
            key=lambda v: json.dumps(v, ensure_ascii=False),
        )
        return [node.type, fixed, moved]
    return [node.type, children]


def render(node, src):
    if node.type not in BODY_TYPES:
        chunks = []
        pos = node.start_byte
        for c in node.children:
            chunks.extend([src[pos : c.start_byte], render(c, src)])
            pos = c.end_byte
        chunks.append(src[pos : node.end_byte])
        return b"".join(chunks)
    members = [c for c in node.named_children if c.type not in COMMENT_TYPES]
    if not members:
        return src[node.start_byte : node.end_byte]
    prefix_end = node.start_byte + (
        1 if src[node.start_byte : node.start_byte + 1] in (b"{", b";") else 0
    )
    pos = prefix_end
    entries = []
    for idx, c in enumerate(members):
        end = c.end_byte
        # A comment on the declaration's closing line belongs to that declaration.
        next_start = (
            members[idx + 1].start_byte if idx + 1 < len(members) else node.end_byte
        )
        gap = src[end:next_start]
        trailing = re.match(rb"[ \t]*//[^\n]*", gap)
        if trailing:
            end += trailing.end()
        leading = src[pos : c.start_byte]
        # Retain indentation for declarations originally sharing a line.
        text = leading + render(c, src) + src[c.end_byte : end]
        if b"\n" not in leading and not leading.strip():
            indent = b" " * (
                c.start_point.column if idx == 0 else members[0].start_point.column
            )
            text = indent + text.lstrip(b" \t")
        text = text.strip(b"\r\n")
        rank = (
            0
            if c.type not in METHOD_TYPES | CONSTRUCTOR_TYPES | NESTED_TYPES
            else (
                1 if c.type in CONSTRUCTOR_TYPES else 3 if c.type in NESTED_TYPES else 2
            )
        )
        name = c.child_by_field_name("name")
        name = src[name.start_byte : name.end_byte].decode() if name else ""
        calls = []

        def visit(n):
            if n.type == "method_invocation":
                nm = n.child_by_field_name("name")
                obj = n.child_by_field_name("object")
                if nm and (
                    obj is None
                    or src[obj.start_byte : obj.end_byte] in (b"this", b"super")
                ):
                    calls.append(src[nm.start_byte : nm.end_byte].decode())
            if n is not c and n.type in NESTED_TYPES:
                return
            for child in n.named_children:
                visit(child)

        if c.type in METHOD_TYPES:
            visit(c)
        entries.append(
            dict(
                node=c,
                text=text,
                rank=rank,
                visibility=visibility(c, src),
                name=name,
                calls=calls,
                index=idx,
            )
        )
        pos = end
    ordered = []
    fixed = [e for e in entries if e["rank"] == 0]

    def is_static(e):
        n = e["node"]
        if n.type == "static_initializer":
            return True
        mod = next((c for c in n.children if c.type == "modifiers"), None)
        return mod is not None and bool(
            re.search(rb"\bstatic\b", src[mod.start_byte : mod.end_byte])
        )

    # Keep the relative order of all static initializers and all instance initializers.
    ordered.extend(e for e in fixed if is_static(e))
    ordered.extend(e for e in fixed if not is_static(e))
    ordered.extend(
        sorted((e for e in entries if e["rank"] == 1), key=lambda e: e["visibility"])
    )
    # Prefer callers before their helpers within each visibility group.
    for vis in range(4):
        group = [e for e in entries if e["rank"] == 2 and e["visibility"] == vis]
        byname = {}
        for e in group:
            byname.setdefault(e["name"], []).append(e)
        seen = set()

        def emit(e):
            if e["index"] in seen:
                return
            seen.add(e["index"])
            ordered.append(e)
            for call in e["calls"]:
                for target in byname.get(call, []):
                    emit(target)

        called = {name for e in group for name in e["calls"] if name != e["name"]}
        for e in group:
            if e["name"] not in called:
                emit(e)
        for e in group:
            emit(e)
    ordered.extend(e for e in entries if e["rank"] == 3)
    if [e["index"] for e in ordered] == list(range(len(entries))):
        # No local movement: preserve all original spacing.
        result = []
        cursor = node.start_byte
        for c in members:
            result.extend([src[cursor : c.start_byte], render(c, src)])
            cursor = c.end_byte
        result.append(src[cursor : node.end_byte])
        return b"".join(result)

    def field_group(e):
        if e["node"].type not in {"field_declaration", "constant_declaration"}:
            return None
        mod = next((c for c in e["node"].children if c.type == "modifiers"), None)
        flags = src[mod.start_byte : mod.end_byte] if mod else b""
        kind = e["node"].child_by_field_name("type")
        scalar = kind is not None and kind.type in {
            "integral_type",
            "floating_point_type",
            "boolean_type",
        }
        return (is_static(e), bool(re.search(rb"\bfinal\b", flags)), scalar)

    chunks = []
    for i, e in enumerate(ordered):
        if i:
            previous = ordered[i - 1]
            same_fields = field_group(e) is not None and field_group(e) == field_group(
                previous
            )
            chunks.append(b"\n" if same_fields else b"\n\n")
        chunks.append(e["text"])
    return (
        src[node.start_byte : prefix_end]
        + b"\n"
        + b"".join(chunks)
        + src[pos : node.end_byte]
    )


def reorder(source):
    """Validate before returning a reordered source; never tolerate parse errors."""
    before = PARSER.parse(source)
    if before.root_node.has_error:
        raise ValueError("Java source contains parse errors")
    result = render(before.root_node, source)
    after = PARSER.parse(result)
    if after.root_node.has_error:
        raise ValueError("Reordered Java source contains parse errors")
    if canonical(before.root_node, source) != canonical(after.root_node, result):
        raise ValueError("Reordering changed content or initializer order")
    # A standard must converge, including overloaded methods and call cycles.
    if render(after.root_node, result) != result:
        raise ValueError("Member order did not converge after one pass")
    return result


def java_files(paths):
    result = set()
    for path in paths:
        if not path.exists():
            raise ValueError("Path does not exist: " + str(path))
        candidates = [path] if path.is_file() else path.rglob("*.java")
        result.update(
            p for p in candidates if p.suffix == ".java" and "target" not in p.parts
        )
    return sorted(result)


def main(argv=None):
    argument_parser = argparse.ArgumentParser(description=__doc__)
    modes = argument_parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--check", action="store_true", help="report unordered files (default)"
    )
    modes.add_argument("--apply", action="store_true", help="write reordered files")
    modes.add_argument("--patch", action="store_true", help="emit an apply_patch patch")
    argument_parser.add_argument(
        "--diff", action="store_true", help="show a unified diff"
    )
    argument_parser.add_argument(
        "paths", nargs="*", type=Path, help="Java files or directories"
    )
    args = argument_parser.parse_args(argv)
    paths = args.paths or [Path(__file__).resolve().parents[1]]
    changes = []
    try:
        # Validate the entire batch before writing any file.
        for path in java_files(paths):
            original = path.read_bytes()
            try:
                ordered = reorder(original)
            except ValueError as error:
                raise ValueError(str(path) + ": " + str(error)) from error
            if original != ordered:
                changes.append((path, original, ordered))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.patch:
        print("*** Begin Patch")
    for path, original, ordered in changes:
        diff = list(
            difflib.unified_diff(
                original.decode().splitlines(True),
                ordered.decode().splitlines(True),
                fromfile=str(path),
                tofile=str(path),
                n=3,
            )
        )
        if args.patch:
            print("*** Update File: " + str(path))
            # Repeated test helpers can have identical short diff contexts.
            # Match the entire original file so patch application stays unambiguous.
            print("@@")
            for line in original.decode().splitlines():
                print("-" + line)
            for line in ordered.decode().splitlines():
                print("+" + line)
        else:
            print(path)
            if args.diff:
                sys.stdout.writelines(diff)
        if args.apply:
            path.write_bytes(ordered)
    if args.patch:
        print("*** End Patch")
    else:
        print(
            f"{len(changes)} file(s) {'reordered' if args.apply else 'need member ordering'}"
        )
    return 0 if args.apply or args.patch or not changes else 1


if __name__ == "__main__":
    raise SystemExit(main())
