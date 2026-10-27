import math

from ..config.values import duration
from .transport import ExecutionError


def _identities(value):
    instances = value['instances']
    ids = [item['id'] for item in instances]
    if len(ids) != len(set(ids)):
        raise ExecutionError('Duplicate instance identity in collection')
    return set(ids)


def compare(actual, expected, spec):
    """Return actual operands as recorded at the gate, plus its aggregate verdict."""
    name = spec['returns'].get('name')
    if name == 'MetricValue':
        if spec.get('unit') and actual['unit'] != spec['unit']:
            raise ExecutionError(f'Metric unit mismatch: {actual["unit"]!r}')
        values = [actual['value']]
    elif name == 'QueryResult':
        values = [sample['value'] for sample in actual['samples']]
        if not values:
            raise ExecutionError('Prometheus query returned no samples')
    elif name == 'PerMasterEngines':
        values = list(actual['instances'].values())
        if not values:
            raise ExecutionError('No Master returned its Engine collection')
    else:
        values = [actual]
    passed = True
    for value in values:
        for operator, target in expected.items():
            if isinstance(target, str) and spec.get('unit') == 's':
                target = float(duration(target))
            operand = value
            if operator == 'count':
                operand = len(_identities(value)) if isinstance(value, dict) else len(value)
                good = operand == target
            elif isinstance(value, dict) and 'instances' in value:
                good = operator == 'eq' and _identities(value) == _identities(target)
            elif operator == 'eq':
                good = value == target
            else:
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ExecutionError('Numeric gate received non-finite/non-numeric value')
                good = value >= target if operator == 'gte' else value <= target
            passed = passed and good
    return passed
