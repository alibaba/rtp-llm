"""Optional horizontal reference lines on existing panels; never evaluate gates."""

import math
from dataclasses import dataclass

from reporting.catalog import PALETTE
from reporting.curves import apply_panel_presets


@dataclass(frozen=True)
class Threshold:
    """Value in the bound curve's source unit; window in the panel's local seconds."""
    value: float
    start: float
    end: float

    def __post_init__(self):
        if (any(type(x) not in (int, float) or not math.isfinite(x)
                for x in (self.value, self.start, self.end)) or self.start >= self.end):
            raise ValueError('gate line requires a finite value and increasing window')


def apply_gate_lines(panels, presentation, thresholds):
    """Case supplies frozen values/windows; YAML chooses where to display them.

    None means the optional gate was not enabled. Unknown keys are configuration
    errors. Observation points, captions and clocks are not modified.
    """
    result = [dict(panel, series=list(panel['series'])) for panel in panels]
    by_id = {panel['id']: panel for panel in result}
    for descriptor in presentation.charts.panels:
        descriptor.validate_bindings(presentation.charts.curves)
        panel = by_id[descriptor.id]
        for binding in descriptor.gate_lines:
            if binding.reference not in thresholds:
                raise ValueError('unknown gate line reference: ' + binding.reference)
            threshold = thresholds[binding.reference]
            if threshold is None:
                continue
            style = presentation.charts.curves[binding.curve_id]
            source = next((s for s in panel['series'] if s['curve_id'] == binding.curve_id), None)
            panel['series'].append(dict(
                curve_id=binding.curve_id + '/gate/' + binding.reference,
                metric_id=style.metric_id, labels=dict(style.labels or ()), scale=style.scale,
                measurement=dict(kind='gate_reference', reference=binding.reference),
                source_type='configuration', role='gate_reference',
                name=binding.name, group='门禁', axis=style.axis,
                unit=style.unit or (source or {}).get('unit', ''),
                color=style.color or (source or {}).get('color') or PALETTE[0],
                dash=[6, 4], hidden=False,
                points=[dict(x=t, y=threshold.value * style.scale)
                        for t in (threshold.start, threshold.end)],
                description='本次运行的窗口参考值；正式门禁以检查结果为准',
            ))
        apply_panel_presets(panel, descriptor)
    return result
