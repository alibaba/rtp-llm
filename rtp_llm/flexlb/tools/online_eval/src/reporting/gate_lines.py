"""Explicit panel bindings project frozen gate references without adjudication."""
from __future__ import annotations
from dataclasses import dataclass
import copy

from analysis.gate_references import validate_reference
from reporting.catalog import TONE_TO_COLOR


@dataclass(frozen=True)
class GateLine:
    check: str
    curve_id: str
    reference: str = 'threshold'
    name: str | None = None
    color: str | None = None
    hidden: bool = False

    @classmethod
    def read(cls, data):
        if (not isinstance(data, dict) or not {'check', 'curve_id'} <= set(data)
                or set(data) - set(cls.__dataclass_fields__)
                or any(type(data.get(k)) is not str or not data[k].strip()
                       for k in ('check', 'curve_id'))
                or any(type(data[k]) is not str or not data[k].strip()
                       for k in ('reference', 'name', 'color') if k in data)
                or 'hidden' in data and type(data['hidden']) is not bool):
            raise ValueError('invalid panel gate_lines binding')
        return cls(**data)

    def to_dict(self):
        return {key: value for key, value in vars(self).items() if value is not None}


def validate_bindings(presentation, definitions):
    catalog = {(d['check'], d['reference']): d for d in definitions}
    for panel in presentation.charts.panels:
        for binding in panel.gate_lines:
            spec = catalog.get((binding.check, binding.reference))
            if spec is None:
                raise ValueError('unknown gate reference: ' + binding.check + '/' + binding.reference)
            style = presentation.charts.curves[binding.curve_id]
            if spec['metric_id'] != style.metric_id or any(
                    dict(style.labels or ()).get(k) != v for k, v in spec['labels'].items()):
                raise ValueError('gate reference metric/population does not match curve: ' + binding.curve_id)
            if style.source_unit != spec['unit']:
                raise ValueError('gate reference unit does not match curve: ' + binding.curve_id)


def apply_gate_lines(spec, presentation, references):
    """Absolute adjudication windows become local coordinates exactly once."""
    catalog = {}
    for value in references:
        validate_reference(value)
        key = (value['check'], value['reference'])
        if key in catalog:
            raise ValueError('duplicate frozen gate reference: ' + '/'.join(key))
        catalog[key] = value
    result = dict(spec, panels=[dict(p, series=list(p['series'])) for p in spec['panels']])
    panels = {p['id']: p for p in result['panels']}
    omitted = []
    for descriptor in presentation.charts.panels:
        descriptor.validate_bindings(presentation.charts.curves)
        panel = panels[descriptor.id]
        diagnostics = []
        for binding in descriptor.gate_lines:
            key = (binding.check, binding.reference)
            value = catalog.get(key)
            if value is None or value['status'] != 'AVAILABLE':
                diagnostics.append(dict(check=binding.check, reference=binding.reference,
                    status='UNAVAILABLE', reason=value['reason'] if value else '门禁未执行或未冻结参考线'))
                continue
            style = presentation.charts.curves[binding.curve_id]
            if value['metric_id'] != style.metric_id or any(
                    dict(style.labels or ()).get(k) != v for k, v in value['labels'].items()):
                raise ValueError('frozen gate reference does not match configured curve')
            if value['unit'] != style.source_unit:
                raise ValueError('frozen gate reference unit mismatch')
            origin = spec['timeOriginEpochS']
            name = binding.name or style.name.format(**dict(style.labels or ())) + ' 门禁线'
            for index, span in enumerate(value['windows']):
                panel['series'].append(dict(
                    curve_id=binding.curve_id + '/gate/' + binding.check + '/' + binding.reference + '/' + str(index),
                    metric_id=style.metric_id, source_type='configuration', role='gate_reference',
                    name=name + ' · ' + span['name'], group='门禁参考线', axis=style.axis,
                    unit=style.unit if style.unit is not None else value['unit'], color=binding.color or style.color or TONE_TO_COLOR['danger'],
                    dash=[6, 4], hidden=binding.hidden,
                    points=[dict(x=span[k] - origin, y=value['value'] * style.scale)
                            for k in ('start_epoch_s', 'end_epoch_s')],
                    description=value['description'] + '；' + value['op'] + ' ' + str(value['value'])
                                + ' ' + value['unit'] + '；窗口 ' + span['name'],
                    provenance=copy.deepcopy(value), measurement=dict(kind='gate_reference',
                        check=binding.check, reference=binding.reference, window=span['name']),
                ))
            diagnostics.append(dict(check=binding.check, reference=binding.reference, status='AVAILABLE'))
        if diagnostics:
            panel['gate_lines'] = diagnostics
            missing = [d['check'] + '/' + d['reference'] + '：' + d['reason']
                       for d in diagnostics if d['status'] != 'AVAILABLE']
            if missing:
                panel['caption'] += ' 门禁参考线不可用：' + '；'.join(missing) + '。'
        if descriptor.requires_gate_evidence and not any(
                point.get('y') is not None for curve in panel['series'] for point in curve['points']):
            omitted.append(dict(panel=descriptor.id, title=descriptor.title,
                                reason='未冻结门禁统计或参考值', references=diagnostics))
        from reporting.curves import apply_panel_presets
        apply_panel_presets(panel, descriptor)
    if omitted:
        from reporting.components import details
        identities = {row['panel'] for row in omitted}
        result['panels'] = [p for p in result['panels'] if p['id'] not in identities]
        result['sections'] = list(result.get('sections', [])) + [details(
            '未执行的门禁图', omitted, identity='gate.reference.availability')]
    return result


def check_references(checks):
    """Keep frozen evidence when flattening the shared run check tree."""
    return [value for check in checks for value in
            (*check.evidence.get('gate_references', []), *check_references(check.children))]
