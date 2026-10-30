from pathlib import Path
import re

import yaml
from yaml.nodes import MappingNode

from .errors import ConfigError


class StrictLoader(yaml.SafeLoader):
    # Keep timestamps and words like 'on' as strings. Only true/false are booleans.
    yaml_implicit_resolvers = {
        key: [(tag, pattern) for tag, pattern in entries if tag not in (
            'tag:yaml.org,2002:timestamp', 'tag:yaml.org,2002:bool')]
        for key, entries in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, MappingNode):
            raise ConfigError('Expected mapping')
        result = {}
        for key_node, value_node in node.value:
            if key_node.tag == 'tag:yaml.org,2002:merge':
                raise ConfigError('YAML merge keys are unsupported')
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError(f'Mapping keys must be strings at line {key_node.start_mark.line + 1}')
            if key in result:
                raise ConfigError(f'Duplicate key {key!r} at line {key_node.start_mark.line + 1}')
            result[key] = self.construct_object(value_node, deep=deep)
        return result


StrictLoader.add_implicit_resolver(
    'tag:yaml.org,2002:bool', re.compile(r'^(?:true|false)$'), list('tf'))


def load_yaml(path: str | Path) -> dict:
    try:
        value = yaml.load(Path(path).read_text(encoding='utf-8'), Loader=StrictLoader)
    except (yaml.YAMLError, OSError, RecursionError) as error:
        raise ConfigError(str(error)) from error
    if not isinstance(value, dict):
        raise ConfigError('Case must be a mapping')
    # Prevent alias cycles from reaching recursive validation or JSON serialization.
    def walk(node, stack):
        if not isinstance(node, (dict, list)):
            return
        if id(node) in stack:
            raise ConfigError('Cyclic YAML aliases are unsupported')
        stack.add(id(node))
        for child in node.values() if isinstance(node, dict) else node:
            walk(child, stack)
        stack.remove(id(node))
    walk(value, set())
    return value
