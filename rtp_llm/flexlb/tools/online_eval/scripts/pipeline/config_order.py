"""Order authored YAML fields without rewriting scalar values, comments or lists."""
import argparse
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, SequenceNode

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_FIELDS = (
    'case_schema_version', 'case', 'program', 'metadata', 'profiles',
    'environment', 'execution', 'parameters', 'parameter_schema',
    'variant_axis', 'variants', 'profile_overrides', 'analysis', 'reports',
)
VIEW_FIELDS = ('report_view_schema_version', 'kind', 'report', 'metrics', 'charts', 'sections')
VIEW_BLOCK_FIELDS = {
    ('report',): ('subtitle', 'id', 'producer'),
    ('metrics',): ('query_plan', 'diagnostic_only'),
    ('charts',): ('time_origin_label', 'events', 'event_ids', 'curves', 'panels', 'group_by',
                  'detail_labels', 'summaries', 'default_visible', 'max_points_per_series', 'presets'),
}
BLOCK_FIELDS = {
    'metadata': ('kind', 'description', 'category', 'tags'),
    'environment': ('backend', 'perf_preset', 'model_override', 'master_layout',
                    'discovery', 'n_prefill', 'n_decode', 'prefill_cache_policy',
                    'config_overrides', 'metric_whitelist'),
    'execution': ('timeout_s', 'stage_timeout_s', 'cleanup_timeout_s', 'collection', 'monitoring'),
    'parameters': ('traffic', 'procedure', 'observation', 'analysis', 'checks'),
    'traffic': ('kind', 'group_id', 'phase_id', 'source', 'client', 'targets', 'duration_s',
             'timeout_ms', 'replay_speed', 'loop', 'max_concurrency', 'max_requests',
             'fallback', 'live_events', 'poll_s', 'jvm_xms', 'jvm_xmx'),
    'source': ('kind', 'model', 'version', 'parameters'),
    'client': ('playback',),
}
# Sampling settings, metric bindings, capture bounds, then windows.
OBSERVATION_FIELDS = (
    'warmup_timeout_s', 'sample_s', 'window_s', 'step_s', 'max_gap_s',
    'inputs', 'capture', 'windows',
)

QUERY_FIELDS = ('promql', 'mode', 'producer', 'source_type', 'unit', 'value_kind',
                'labels', 'calculation', 'measurement', 'exported_metrics', 'required')
CURVE_FIELDS = ('metric_id', 'labels', 'name', 'group', 'unit', 'axis', 'scale',
                'color', 'hidden')


def fields(kind, path):
    if not path:
        return {'scenarios': SCENARIO_FIELDS, 'report_views': VIEW_FIELDS,
                'monitoring': ('metric_plan_schema_version', 'include', 'exclude', 'sources', 'produced'),
                'suite': ('suite_schema_version', 'default_suite', 'ci_suites'),
                'mode': ('mode_profiles_schema_version', 'runtime_modes', 'master_modes')}[kind]
    if kind == 'monitoring' and (len(path) == 3 and path[0] == 'sources'
                                 or len(path) == 2 and path[0] == 'produced'):
        return QUERY_FIELDS
    if kind == 'monitoring' and len(path) == 3 and path[0] == 'produced' and path[-1] == 'calculation':
        return ('calculator', 'window', 'selection', 'token_field', 'field', 'percentile', 'bucket_s')
    if kind == 'report_views':
        if path in VIEW_BLOCK_FIELDS:
            return VIEW_BLOCK_FIELDS[path]
        if len(path) == 3 and path[:2] == ('charts', 'curves'):
            return CURVE_FIELDS
        if len(path) == 2 and path[0] == 'sections':
            return ('title', 'columns', 'opened')
    if kind == 'scenarios' and path in {
        ('parameters', 'observation'), ('variants', '[]', 'parameters', 'observation'),
    }:
        return OBSERVATION_FIELDS
    # Trace/source.parameters is a producer-specific contract, not case.parameters.
    if kind == 'scenarios' and path[-1] in BLOCK_FIELDS and (
        path[-1] != 'parameters' or len(path) == 1
    ):
        return BLOCK_FIELDS[path[-1]]
    return ()


def ordered_yaml(text, kind):
    lines = text.splitlines(keepends=True)
    node = yaml.compose(text)

    def render(node, path):
        if isinstance(node, SequenceNode):
            for item in node.value:
                render(item, path + ('[]',))
            return
        if not isinstance(node, MappingNode):
            return
        for key, value in node.value:
            render(value, path + (key.value,))
        order = fields(kind, path)
        if not order or node.flow_style or not node.value:
            return
        first = node.value[0][0].start_mark
        # A sequence-item mapping begins after '- '; leave its layout untouched.
        if lines[first.line][:first.column].strip():
            return
        starts = []
        for key, _ in node.value:
            line = key.start_mark.line
            while line > first.line and (not lines[line-1].strip()
                                         or lines[line-1].lstrip().startswith('#')):
                line -= 1
            starts.append(line)
        end = node.end_mark.line + (node.end_mark.column > first.column)
        chunks = [(key.value, lines[start:stop]) for (key, _), start, stop in
                  zip(node.value, starts, starts[1:] + [end])]
        rank = {name: i for i, name in enumerate(order)}
        chunks.sort(key=lambda item: rank.get(item[0], len(order)))
        replacement = [line for _, chunk in chunks for line in chunk]
        lines[starts[0]:end] = replacement

    render(node, ())
    return ''.join(lines)


def configurations():
    for kind in ('scenarios', 'monitoring', 'report_views'):
        for path in sorted((ROOT/'config'/kind).glob('*.yaml')):
            yield path, kind
    yield ROOT/'config/suites.yaml', 'suite'
    yield ROOT/'mode_profiles.yaml', 'mode'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='check without rewriting files')
    args = parser.parse_args(argv)
    changed = []
    for path, kind in configurations():
        original = path.read_text()
        formatted = ordered_yaml(original, kind)
        if yaml.safe_load(original) != yaml.safe_load(formatted):
            raise ValueError(f"field ordering changed YAML values: {path}")
        if original != formatted:
            changed.append(path)
            if not args.check:
                path.write_text(formatted)
    for path in changed:
        print(path.relative_to(ROOT))
    return 1 if changed and args.check else 0
