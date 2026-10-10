"""YAML reading order preserves authored data, comments and list semantics."""
import yaml

from scripts.pipeline.config_order import configurations, main, ordered_yaml


def test_order_preserves_scalar_spelling_comments_and_lists():
    source = '''reports:
- custom.yaml
parameters:
  flow:
    timeout_ms: 10 # budget
    # describes the chosen source
    source:
      kind: trace
      parameters:
        quoted: 'false'
        multiline: |
          first
          second
    targets: [B, A]
    loop: false
case_schema_version: 2
case: sample
program: default
variants:
- id: later
  parameters: {threshold: 1.0}
- id: earlier
  parameters: {threshold: 0}
'''
    result = ordered_yaml(source, 'scenarios')
    assert yaml.safe_load(result) == yaml.safe_load(source)
    assert list(yaml.safe_load(result)) == ['case_schema_version', 'case', 'program', 'parameters', 'variants', 'reports']
    assert "quoted: 'false'" in result
    assert 'timeout_ms: 10 # budget' in result
    assert '# describes the chosen source\n    source:' in result
    assert 'targets: [B, A]' in result
    assert ordered_yaml(result, 'scenarios') == result


def test_nested_mapping_does_not_take_fields_from_its_next_sibling():
    source = '''metric_plan_schema_version: 3
sources:
  mock:
    one:
      promql: one${selector}
      required: true
      labels: []
    two:
      promql: two${selector}
      labels: []
'''
    result = ordered_yaml(source, 'monitoring')
    assert yaml.safe_load(result) == yaml.safe_load(source)
    assert list(yaml.safe_load(result)['sources']['mock']['one']) == ['promql', 'labels', 'required']
    assert 'required' not in yaml.safe_load(result)['sources']['mock']['two']
    assert ordered_yaml(result, 'monitoring') == result


def test_named_collections_keep_their_order():
    source = """charts:
  curves:
    second:
      name: two
      metric_id: mock/two
      labels: {}
    first:
      name: one
      metric_id: mock/one
      labels: {}
  panels:
  - id: second
    curve_ids: [second, first]
kind: selected
report_view_schema_version: 1
report:
  subtitle: selected
"""
    result = ordered_yaml(source, 'report_views')
    parsed = yaml.safe_load(result)
    assert parsed == yaml.safe_load(source)
    assert list(parsed['charts']['curves']) == ['second', 'first']
    assert parsed['charts']['panels'][0]['curve_ids'] == ['second', 'first']
    assert list(parsed['charts']['curves']['second']) == ['metric_id', 'labels', 'name']


def test_bundled_configs_follow_declared_order():
    for path, kind in configurations():
        assert path.read_text() == ordered_yaml(path.read_text(), kind), path
    assert main(['--check']) == 0


def test_observation_order_covers_base_and_variant_without_sorting_named_windows():
    source = """parameters:
  observation:
    windows:
      second: {until: {event: end, offset_s: 10}}
      first: {from: {event: start, offset_s: 0}}
    capture: {max_samples: 10, max_bytes: 1024}
    collapse: {sustain_s: 1}
    slo: {ttft_ms: 2}
    inputs: {engine: {source: metric_store}}
    max_gap_s: 3
    sample_s: 1
variants:
- id: extra
  parameters:
    observation:
      windows: {measurement: {}}
      capture: {max_samples: 20}
      sample_s: 2
"""
    result = ordered_yaml(source, 'scenarios')
    parsed = yaml.safe_load(result)
    assert parsed == yaml.safe_load(source)
    assert list(parsed['parameters']['observation']) == [
        'sample_s', 'max_gap_s', 'inputs', 'slo', 'collapse', 'capture', 'windows',
    ]
    assert list(parsed['parameters']['observation']['windows']) == ['second', 'first']
    assert list(parsed['variants'][0]['parameters']['observation']) == ['sample_s', 'capture', 'windows']
    assert ordered_yaml(result, 'scenarios') == result


def test_observation_order_does_not_rewrite_a_traffic_producers_parameters():
    source = """parameters:
  traffic:
    source:
      parameters:
        observation:
          windows: {}
          sample_s: 1
"""
    result = ordered_yaml(source, 'scenarios')
    assert list(yaml.safe_load(result)['parameters']['traffic']['source']['parameters']['observation']) == ['windows', 'sample_s']
