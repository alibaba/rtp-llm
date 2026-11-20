"""Find unused Java imports, fields, production methods and private constructors in FlexLB.

This is a conservative source audit, not a Java type checker. Method references
are indexed across all modules, including tests. Ambiguous names are retained;
framework callbacks and generated accessors are not treated as ordinary calls.
Private constructors are matched by simple type name and argument count.
Constructor references, potential reflection and generated calls are retained.
A sole no-arg private constructor also controls default constructor visibility.
The commit hook audits the entire source tree, including methods orphaned by
deleting their last caller. It only reports findings; it never edits sources.
"""

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from order_java_members import PARSER

TYPES = {
    "class_declaration",
    "interface_declaration",
    "record_declaration",
    "enum_declaration",
    "annotation_type_declaration",
}
CALLBACKS = {
    "Override",
    "Bean",
    "Autowired",
    "PostConstruct",
    "PreDestroy",
    "Scheduled",
    "EventListener",
    "GetMapping",
    "PostMapping",
    "PutMapping",
    "DeleteMapping",
    "PatchMapping",
    "RequestMapping",
    "JsonCreator",
    "JsonValue",
    "JsonProperty",
    "JsonSetter",
    "JsonGetter",
    "JsonAnyGetter",
    "JsonAnySetter",
    "SpyBean",
    "MockBean",
    "SystemStub",
    "TempDir",
    "Resource",
    "Inject",
    "Value",
}
CONSTRUCTOR_GENERATORS = {"Builder", "SuperBuilder", "Jacksonized"}
GENERATED_ACCESSORS = {"Data", "Getter", "Setter", "Value", "Builder", "SuperBuilder"}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    symbol: str
    rule: str

    def __str__(self):
        return f"{self.path}:{self.line}: {self.rule}: {self.symbol}"


def walk(node):
    yield node
    for child in node.named_children:
        yield from walk(child)


def text(source, node):
    return source[node.start_byte : node.end_byte].decode() if node else ""


def annotations(source, node):
    modifiers = next((c for c in node.named_children if c.type == "modifiers"), None)
    return set(re.findall(r"@(?:[\w$]+\.)*([\w$]+)", text(source, modifiers)))


def owner(node):
    current = node.parent
    while current and current.type not in TYPES:
        current = current.parent
    return current


def symbol(source, node):
    names = [text(source, node.child_by_field_name("name"))]
    current = owner(node)
    while current:
        names.append(text(source, current.child_by_field_name("name")))
        current = owner(current)
    return ".".join(reversed(names))


def production(path):
    return "/src/main/" in f"/{path}" or path.startswith("tools/")


def simple_type_name(source, node):
    if node is None:
        return ""
    if node.type in {"generic_type", "superclass"}:
        return simple_type_name(source, node.named_children[0])
    if node.type in {"scoped_type_identifier", "field_access"}:
        return simple_type_name(source, node.named_children[-1])
    return text(source, node)


def count_arguments(node):
    return sum(
        child.type not in {"line_comment", "block_comment"}
        for child in node.named_children
    )


def audit(sources):
    """Accept relative paths and bytes so tests and Git snapshots use identical rules."""
    parsed = {}
    calls = defaultdict(set)
    constructor_calls = defaultdict(set)
    indirect_construction = set()
    strings = defaultdict(set)
    findings = []
    for path, source in sources.items():
        tree = PARSER.parse(source)
        if tree.root_node.has_error:
            raise ValueError(f"{path}: cannot audit invalid Java syntax")
        nodes = list(walk(tree.root_node))
        parsed[path] = source, nodes
        for node in nodes:
            if node.type == "object_creation_expression":
                name = simple_type_name(source, node.child_by_field_name("type"))
                arity = count_arguments(node.child_by_field_name("arguments"))
                constructor_calls[name].add((path, arity))
            elif node.type == "explicit_constructor_invocation":
                container = owner(node)
                target = text(source, node.child_by_field_name("constructor"))
                if target == "this":
                    name = text(source, container.child_by_field_name("name"))
                else:
                    name = simple_type_name(
                        source, container.child_by_field_name("superclass")
                    )
                arity = count_arguments(node.child_by_field_name("arguments"))
                constructor_calls[name].add((path, arity))
            elif node.type == "class_declaration":
                # A subclass can invoke a no-arg superclass constructor implicitly.
                superclass = simple_type_name(
                    source, node.child_by_field_name("superclass")
                )
                if superclass:
                    constructor_calls[superclass].add((path, 0))
            elif node.type == "class_literal":
                # Class literals can pass the type to reflection or a framework.
                indirect_construction.add(
                    simple_type_name(source, node.named_children[0])
                )
            elif node.type == "method_invocation":
                name = text(source, node.child_by_field_name("name"))
                enclosing = node.parent
                while enclosing and enclosing.type != "method_declaration":
                    enclosing = enclosing.parent
                receiver = text(source, node.child_by_field_name("object"))
                if (
                    receiver not in {"", "this"}
                    or not enclosing
                    or text(source, enclosing.child_by_field_name("name")) != name
                ):
                    calls[name].add(path)
            elif node.type == "method_reference":
                if node.children[-1].type == "new":
                    indirect_construction.add(
                        simple_type_name(source, node.named_children[0])
                    )
                    continue
                # The final named child is the method identifier (or type arguments).
                names = [c for c in node.named_children if c.type == "identifier"]
                if names:
                    calls[text(source, names[-1])].add(path)
            elif node.type == "string_literal":
                value = text(source, node)[1:-1]
                indirect_construction.add(re.split(r"[.$]", value)[-1])
                if production(path):
                    strings[value].add(path)

    for path, (source, nodes) in parsed.items():
        imports = [n for n in nodes if n.type == "import_declaration"]
        code_nodes = [
            n
            for n in nodes
            if not any(i.start_byte <= n.start_byte < i.end_byte for i in imports)
        ]
        names = Counter(
            text(source, n)
            for n in code_nodes
            if n.type in {"identifier", "type_identifier"}
        )
        # Imports referenced by Javadoc links are meaningful to javadoc/doclint.
        docs = "\n".join(text(source, n) for n in nodes if n.type == "block_comment")
        for node in imports:
            declaration = text(source, node)
            if "*" in declaration:
                continue
            name = declaration.rstrip("; ").rsplit(".", 1)[-1]
            if not names[name] and not re.search(r"\b" + re.escape(name) + r"\b", docs):
                findings.append(
                    Finding(path, node.start_point.row + 1, name, "unused-import")
                )

        for node in nodes:
            if node.type == "field_declaration":
                container = owner(node)
                field_annotations = annotations(source, node)
                class_annotations = (
                    annotations(source, container) if container else set()
                )
                if (field_annotations | class_annotations) & (
                    GENERATED_ACCESSORS | CALLBACKS
                ):
                    continue
                modifiers = text(
                    source,
                    next((c for c in node.children if c.type == "modifiers"), None),
                )
                # Public/package fields may be used through inheritance or external protocols.
                if "private" not in modifiers:
                    continue
                for variable in (
                    c for c in node.named_children if c.type == "variable_declarator"
                ):
                    name = text(source, variable.child_by_field_name("name"))
                    if name == "serialVersionUID" or name in strings:
                        continue
                    if names[name] == 1:
                        findings.append(
                            Finding(
                                path,
                                variable.start_point.row + 1,
                                symbol(source, variable),
                                "unused-private-field",
                            )
                        )

            if node.type == "constructor_declaration" and production(path):
                container = owner(node)
                modifiers = next(
                    (c for c in node.named_children if c.type == "modifiers"), None
                )
                name = text(source, node.child_by_field_name("name"))
                if (
                    container.type != "class_declaration"
                    or not modifiers
                    or not any(child.type == "private" for child in modifiers.children)
                    or (annotations(source, node) | annotations(source, container))
                    & (CALLBACKS | CONSTRUCTOR_GENERATORS)
                    or name in indirect_construction
                ):
                    continue
                parameters = node.child_by_field_name("parameters")
                arity = sum(
                    child.type in {"formal_parameter", "spread_parameter"}
                    for child in parameters.named_children
                )
                varargs = any(
                    child.type == "spread_parameter"
                    for child in parameters.named_children
                )
                if (
                    arity == 0
                    and sum(
                        member.type == "constructor_declaration"
                        for member in node.parent.named_children
                    )
                    == 1
                ):
                    continue
                # Direct access to a private constructor stays within its source file.
                if not any(
                    caller == path
                    and (arguments == arity or (varargs and arguments >= arity - 1))
                    for caller, arguments in constructor_calls[name]
                ):
                    findings.append(
                        Finding(
                            path,
                            node.start_point.row + 1,
                            f"{symbol(source, node)}/{arity}",
                            "unused-private-constructor",
                        )
                    )
                continue

            if node.type != "method_declaration" or not production(path):
                continue
            name = text(source, node.child_by_field_name("name"))
            if (
                annotations(source, node) & CALLBACKS
                or node.child_by_field_name("body") is None
            ):
                continue
            if name == "main" and re.search(
                r"\bstatic\s+void\s+main\s*\(\s*String", text(source, node)
            ):
                continue
            # Same-named overloads, inherited implementations and reflection require review.
            if name in strings:
                continue
            references = calls[name]
            if any(production(p) for p in references):
                continue
            rule = (
                "test-only-production-method"
                if references
                else "unused-production-method"
            )
            findings.append(
                Finding(path, node.start_point.row + 1, symbol(source, node), rule)
            )
    return sorted(findings, key=lambda f: (f.path, f.line, f.rule))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="read-only full-tree check (default)"
    )
    parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    sources = {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*.java")
        if "target" not in p.relative_to(root).parts
    }
    try:
        findings = audit(sources)
        print(f"Audited {len(sources)} Java files; {len(findings)} findings.")
        for finding in findings:
            print(finding)
        return int(bool(findings))
    except (ValueError, OSError) as error:
        print(error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
