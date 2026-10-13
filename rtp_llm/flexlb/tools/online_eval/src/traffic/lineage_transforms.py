"""Content projection of decoded prefix lineages; pacing belongs to the client."""


from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import hashlib
import json
from pathlib import Path

from traffic.contracts import PRIORITY, JAVA_LENGTH

from traffic.lineage_model import LineageExpander, LineageModel
from traffic.output_sampling import output_sampler, output_semantics


@dataclass(frozen=True)
class LineageProjection:
    path: str
    count: int
    output_tokens: int
    priority: int
    max_input_tokens: int | None = None
    output_distribution: dict | None = None

    @classmethod
    def read(cls, value):
        required = {"path", "count", "output_tokens", "priority"}
        optional = {"max_input_tokens", "output_distribution"}
        if not isinstance(value, dict) or not required <= set(value) or set(value) - required - optional:
            raise ValueError("lineage requires model path, count and output settings")
        result = cls(**value)
        if (type(result.path) is not str
                or type(result.count) is not int or result.count < 1):
            raise ValueError("invalid lineage path/count")
        cap = result.max_input_tokens
        if cap is not None and (type(cap) is not int or not 1 <= cap <= 2147483647):
            raise ValueError("invalid input length filter")
        if (type(result.output_tokens) is not int or not JAVA_LENGTH.minimum <= result.output_tokens <= JAVA_LENGTH.maximum
                or type(result.priority) is not int or not PRIORITY.minimum <= result.priority <= PRIORITY.maximum):
            raise ValueError("invalid output length/priority")
        return result

    def to_dict(self):
        result = dict(path=self.path, count=self.count,
                      output_tokens=self.output_tokens, priority=self.priority)
        if self.max_input_tokens is not None:
            result["max_input_tokens"] = self.max_input_tokens
        if self.output_distribution is not None:
            result["output_distribution"] = self.output_distribution
        return result


class LineageProjector:
    def __init__(self, decode: Callable[[bytes], LineageModel]):
        self.decode = decode

    def write_trace(self, path, parameters, namespace, base_dir, *, max_requests=None):
        return self.project(path, LineageProjection.read(parameters), namespace, base_dir,
                            max_requests=max_requests)

    def project(self, path, parameters: LineageProjection, namespace, base_dir, *, max_requests=None) -> dict:
        sampler = output_sampler(parameters.to_dict())
        cap = parameters.max_input_tokens
        raw = (Path(base_dir) / parameters.path).read_bytes()
        source_sha256 = hashlib.sha256(raw).hexdigest()
        model = self.decode(raw)
        metadata = model.metadata
        block_size = metadata.block_size
        if type(parameters.count) is not int or len(model.events) != parameters.count:
            raise ValueError('lineage model count mismatch')
        selected = excluded = 0
        with Path(path).open('w') as out:
            for request in LineageExpander(model):
                i, ts, length, labels = (request.source_index, request.timestamp_ms,
                                         request.input_tokens, request.block_labels)
                if cap is not None and length > cap:
                    excluded += 1
                    continue
                out.write(json.dumps(dict(rid=f'{namespace}:{i}', ts=ts, il=length,
                    ol=sampler(i), priority=parameters.priority, cache_key_block_size=block_size,
                    input_token_blocks=list(labels)), separators=(',', ':')) + '\n')
                selected += 1
                if max_requests is not None and selected >= max_requests:
                    break
        if not selected:
            raise ValueError('empty input length selection')
        transformations = []
        if cap is not None:
            transformations.append(dict(kind='input_length_filter', max_input_tokens=cap,
                selected=selected, excluded_before_limit=excluded))
        if parameters.output_distribution:
            transformations.append(dict(kind='output_distribution', parameters=parameters.output_distribution,
                affected_requests=selected))
        if max_requests is not None:
            transformations.append(dict(kind='request_limit', limit=max_requests,
                selected=selected, excluded_after_limit=max(0, len(model.events)-selected-excluded)))
        result = dict(source_codec=dict(name="prefix_lineage", version=metadata.version),
            realism=metadata.realism, tail=metadata.tail.value, arrival=metadata.arrival,
            model_sha256=source_sha256, provenance=metadata.provenance,
            transformations=dict(source_sha256=source_sha256, applied=transformations,
                source_requests=len(model.events), selected_requests=selected))
        if cap is not None:
            result['length_filter'] = dict(max_input_tokens=cap, selected=selected,
                excluded_before_limit=excluded)
        if metadata.version == 2:
            result['token_adjustment'] = metadata.token_adjustment
            if parameters.output_distribution:
                result.update(output_semantics=output_semantics(parameters.to_dict()),
                    output_distribution=parameters.output_distribution, output_cap=parameters.output_tokens)
        else:
            if cap is None:
                result['length_filter'] = dict(max_input_tokens=2147483647,
                    selected=selected, excluded_before_limit=excluded)
            result.update(output_semantics=output_semantics(parameters.to_dict()),
                output_distribution=parameters.output_distribution, output_cap=parameters.output_tokens)
        return result


def write_trace(path, parameters, namespace, base_dir, *, decode, max_requests=None):
    return LineageProjector(decode).write_trace(path, parameters, namespace, base_dir,
                                               max_requests=max_requests)
