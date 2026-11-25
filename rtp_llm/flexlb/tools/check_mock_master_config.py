"""Check the JSON block scalars in Master configuration files."""

import json
import sys
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, ScalarNode


def check_master_config(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    document = yaml.compose(source, Loader=yaml.SafeLoader)
    if not isinstance(document, MappingNode):
        return ["expected a YAML mapping"]

    fields = {
        key.value: (key, value)
        for key, value in document.value
        if isinstance(key, ScalarNode)
    }
    lines = source.splitlines()
    errors = []
    for field in ("FLEXLB_CONFIG", "MODEL_SERVICE_CONFIG"):
        key, node = fields.get(field, (None, None))
        if (
            key is None
            or not isinstance(node, ScalarNode)
            or node.style != "|"
            or key.start_mark.line != node.start_mark.line
        ):
            errors.append(f"{field}: use a YAML literal block (|-)")
            continue
        header = lines[node.start_mark.line][node.start_mark.column :]
        if header.partition("#")[0].strip() != "|-":
            errors.append(f"{field}: use a YAML literal block (|-)")
            continue

        try:
            value = json.loads(node.value)
            formatted = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        except ValueError as error:
            errors.append(f"{field}: invalid JSON: {error}")
            continue

        if not isinstance(value, dict):
            errors.append(f"{field}: expected a JSON object")
            continue
        content = [line.rstrip() for line in node.value.splitlines() if line.strip()]
        if content != formatted.splitlines():
            errors.append(f"{field}: expand JSON with two-space indentation")
    return errors


def main() -> int:
    failed = False
    for filename in sys.argv[1:]:
        try:
            errors = check_master_config(Path(filename))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            errors = [str(error)]
        for error in errors:
            print(f"{filename}: {error}", file=sys.stderr)
        failed |= bool(errors)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
