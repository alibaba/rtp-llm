"""Content projection of decoded prefix lineages; pacing belongs to the client."""

import hashlib
import json
from pathlib import Path

from traffic.capture_contract import BLOCK_SIZE as BLOCK
from traffic.output_sampling import output_sampler, output_semantics


def _expanded(metadata, events):
    if metadata['version'] == 2:
        from traffic.prefix_lineage import expand
        for index, (ts, labels) in enumerate(expand(events)):
            yield index, ts, len(labels) * BLOCK, labels
        return
    paths, next_label = [], 1
    for index, (ts, length, parent, shared) in enumerate(events):
        labels = paths[parent][:shared] if parent >= 0 else []
        fresh = (length + BLOCK - 1) // BLOCK - shared
        if next_label + fresh > 2147483647:
            raise ValueError('lineage label budget exceeded')
        labels.extend(range(next_label, next_label + fresh))
        next_label += fresh
        paths.append(labels)
        yield index, ts, length, labels


def write_trace(path, parameters, namespace, base_dir, *, decode, max_requests=None):
    p = parameters
    required = {'path', 'sha256', 'count', 'output_tokens', 'priority'}
    optional = {'max_input_tokens', 'output_distribution'}
    if not isinstance(p, dict) or not required <= set(p) or set(p) - required - optional:
        raise ValueError('lineage requires pinned model and output settings')
    cap = p.get('max_input_tokens')
    if cap is not None and (type(cap) is not int or not 1 <= cap <= 2147483647):
        raise ValueError('invalid input length filter')
    if type(p['output_tokens']) is not int or not 1 <= p['output_tokens'] <= 2147483647 or type(p['priority']) is not int or not 1 <= p['priority'] <= 100:
        raise ValueError('invalid output length/priority')
    sampler = output_sampler(p)
    raw = (Path(base_dir) / p['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != p['sha256']:
        raise ValueError('lineage model checksum mismatch')
    metadata, events = decode(raw)
    if type(p['count']) is not int or len(events) != p['count']:
        raise ValueError('lineage model count mismatch')
    selected = excluded = 0
    with Path(path).open('w') as out:
        for i, ts, length, labels in _expanded(metadata, events):
            if cap is not None and length > cap:
                excluded += 1
                continue
            out.write(json.dumps(dict(rid=f'{namespace}:{i}', ts=ts, il=length,
                ol=sampler(i), priority=p['priority'], cache_key_block_size=BLOCK,
                input_token_blocks=labels), separators=(',', ':')) + '\n')
            selected += 1
            if max_requests is not None and selected >= max_requests:
                break
    if not selected:
        raise ValueError('empty input length selection')
    transformations = []
    if cap is not None:
        transformations.append(dict(kind='input_length_filter', max_input_tokens=cap,
            selected=selected, excluded_before_limit=excluded))
    if p.get('output_distribution'):
        transformations.append(dict(kind='output_distribution', parameters=p['output_distribution'],
            affected_requests=selected))
    if max_requests is not None:
        transformations.append(dict(kind='request_limit', limit=max_requests,
            selected=selected, excluded_after_limit=max(0, len(events)-selected-excluded)))
    result = dict(realism=metadata['realism'], tail=metadata['tail'], arrival=metadata['arrival'],
        model_sha256=p['sha256'], provenance=metadata['provenance'],
        transformations=dict(source_sha256=p['sha256'], applied=transformations,
            source_requests=len(events), selected_requests=selected))
    if cap is not None:
        result['length_filter'] = dict(max_input_tokens=cap, selected=selected,
            excluded_before_limit=excluded)
    if metadata['version'] == 2:
        result['token_adjustment'] = metadata['token_adjustment']
        if p.get('output_distribution'):
            result.update(output_semantics=output_semantics(p),
                output_distribution=p['output_distribution'], output_cap=p['output_tokens'])
    else:
        if cap is None:
            result['length_filter'] = dict(max_input_tokens=2147483647,
                selected=selected, excluded_before_limit=excluded)
        result.update(output_semantics=output_semantics(p),
            output_distribution=p.get('output_distribution'), output_cap=p['output_tokens'])
    return result
