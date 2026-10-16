"""Cross data archives with behavior variants before the ordinary case compiler."""
from dataclasses import replace

from document_io import DocumentError as ScenarioError
from monitoring.identity import NAME
from scenario.config_data import mapping, merge_data, merge_environment


def archive_variants(config, variants, source):
    """Archive data precedes behavior patches; the default archive keeps public IDs."""
    archives = config.data_archives
    if archives is None:
        return [(variant, None, variant.identity) for variant in variants]
    path = source + '.data_archives'
    mapping(archives, {'default', 'entries'}, path, required={'default', 'entries'})
    default = archives['default']
    entries = archives['entries']
    if not isinstance(entries, dict) or not entries:
        raise ScenarioError(path + '.entries: expected nonempty mapping')
    for name in (default, *entries):
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ScenarioError(path + ': invalid archive id')
    if default not in entries:
        raise ScenarioError(path + ': default archive must name an entry')
    result = []
    for name in (default, *(name for name in entries if name != default)):
        patch = entries[name]
        location = path + '.entries.' + name
        mapping(patch, {'environment', 'parameters', 'parameter_schema'}, location)
        for field, value in patch.items():
            if not isinstance(value, dict):
                raise ScenarioError(location + '.' + field + ': expected mapping')
        for variant in variants:
            identity = (variant.identity if name == default else
                        name if variant.identity == config.default_variant else name + '__' + variant.identity)
            row = replace(variant, identity=identity,
                environment=merge_environment(patch.get('environment', {}), variant.environment),
                parameters=merge_data(patch.get('parameters', {}), variant.parameters),
                parameter_schema=merge_data(patch.get('parameter_schema', {}), variant.parameter_schema))
            result.append((row, name, variant.identity))
    return result
