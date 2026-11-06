"""Exact-token cache-prefill prompt and shared-seed construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _encode(tokenizer: Any, text: str) -> List[int]:
    return list(tokenizer.encode(text))


class _FastConstructionUnavailable(Exception):
    pass


@dataclass
class CasePrompts:
    seed_text: str
    prefix_ids: List[int] = field(default_factory=list)
    run_texts: List[str] = field(default_factory=list)
    run_ids: List[List[int]] = field(default_factory=list)
    built_len: int = 0
    run_specs: Optional[List[Dict[str, Any]]] = None


class PrefixPromptFactory:
    """Build exact-length prompts and verify their shared token prefix."""

    SHARED_PREFIX_MARKER = "cache_grid_shared_prefix_"

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self._word = " hello"
        self._filler_ids = _encode(tokenizer, self._word)
        self._fast_path = len(self._filler_ids) == 1
        self._last_exact_key = None
        self._last_exact_ids = None
        self._last_exact_text = None
        self._marker_ids_cache: Dict[str, Optional[List[int]]] = {}
        self._tail_ids_cache: Dict[str, Optional[List[int]]] = {}
        self._verified_shapes = set()
        self._unsafe_shapes = set()

    def _store_exact(self, key: Any, text: str, ids: List[int]):
        self._last_exact_key = key
        self._last_exact_text = text
        self._last_exact_ids = ids
        return text, ids

    def _exact_text_and_ids(self, target_len: int, prefix: str):
        if target_len <= 0:
            raise ValueError(f"input length must be positive, got {target_len}")
        key = (int(target_len), prefix)
        if self._last_exact_key == key and self._last_exact_ids is not None:
            return self._last_exact_text, self._last_exact_ids
        if self._fast_path:
            prefix_ids = _encode(self.tokenizer, prefix)
            needed = target_len - len(prefix_ids)
            if needed >= 0:
                candidate = prefix + self._word * needed
                candidate_ids = _encode(self.tokenizer, candidate)
                if len(candidate_ids) == target_len:
                    return self._store_exact(key, candidate, candidate_ids)
        repeats = 1
        while len(_encode(self.tokenizer, prefix + self._word * repeats)) < target_len:
            repeats *= 2
        lo, hi = 0, repeats
        while lo < hi:
            mid = (lo + hi) // 2
            if len(_encode(self.tokenizer, prefix + self._word * mid)) < target_len:
                lo = mid + 1
            else:
                hi = mid
        candidate = prefix + self._word * lo
        candidate_ids = _encode(self.tokenizer, candidate)
        if len(candidate_ids) == target_len:
            return self._store_exact(key, candidate, candidate_ids)
        lo, hi = 0, len(candidate)
        best = prefix
        best_ids = _encode(self.tokenizer, best)
        while lo <= hi:
            mid = (lo + hi) // 2
            current = candidate[:mid]
            current_ids = _encode(self.tokenizer, current)
            if len(current_ids) <= target_len:
                if len(current_ids) == target_len:
                    return self._store_exact(key, current, current_ids)
                best, best_ids = current, current_ids
                lo = mid + 1
            else:
                hi = mid - 1
        if len(best_ids) != target_len:
            raise ValueError(
                f"cannot construct exact tokenizer length {target_len}; got {len(best_ids)}"
            )
        return self._store_exact(key, best, best_ids)

    def make_case(
        self, case_id: Any, total_len: int, cache_len: int
    ) -> Tuple[str, str, int]:
        if cache_len < 0 or cache_len >= total_len:
            raise ValueError(
                "cache_len must satisfy 0 <= cache_len < total_len, "
                f"got {cache_len}/{total_len}"
            )
        marker = f"{self.SHARED_PREFIX_MARKER}{case_id}_"
        if cache_len == 0:
            target, ids = self._exact_text_and_ids(total_len, marker)
            return target, "", len(ids)
        prefix, prefix_ids = self._exact_text_and_ids(cache_len, marker)
        target, target_ids = self._exact_text_and_ids(total_len, prefix + " __suffix_")
        if len(target_ids) != total_len or target_ids[:cache_len] != prefix_ids:
            target, target_ids = self._exact_text_and_ids(total_len, prefix)
        if len(target_ids) != total_len or target_ids[:cache_len] != prefix_ids:
            raise ValueError(f"unable to preserve exact prefix for case={case_id}")
        return target, prefix, len(target_ids)

    def make_seed_and_ids(
        self,
        case_id: Any,
        prefix: str,
        cache_len: int,
        commit_tail_tokens: int,
    ) -> Tuple[str, List[int]]:
        if cache_len <= 0:
            return "", []
        seed_len = cache_len + commit_tail_tokens
        seed, seed_ids = self._exact_text_and_ids(
            seed_len, prefix + f" __seed_commit_{case_id}_"
        )
        prefix_ids = _encode(self.tokenizer, prefix)
        if len(prefix_ids) != cache_len or seed_ids[:cache_len] != prefix_ids:
            seed, seed_ids = self._exact_text_and_ids(seed_len, prefix)
        if len(seed_ids) != seed_len or seed_ids[:cache_len] != prefix_ids:
            raise ValueError(f"unable to build cache seed for case={case_id}")
        return seed, seed_ids

    def make_seed(
        self, case_id: Any, prefix: str, cache_len: int, commit_tail_tokens: int
    ) -> str:
        return self.make_seed_and_ids(case_id, prefix, cache_len, commit_tail_tokens)[0]

    def _safe_marker_ids(self, marker: str) -> Optional[List[int]]:
        if not self._fast_path:
            return None
        if marker in self._marker_ids_cache:
            return self._marker_ids_cache[marker]
        marker_ids = _encode(self.tokenizer, marker)
        probe = _encode(self.tokenizer, marker + self._word * 8)
        result = marker_ids if probe == marker_ids + self._filler_ids * 8 else None
        self._marker_ids_cache[marker] = result
        return result

    def _safe_tail_ids(self, tail: str) -> Optional[List[int]]:
        if not self._fast_path:
            return None
        if tail in self._tail_ids_cache:
            return self._tail_ids_cache[tail]
        tail_ids = _encode(self.tokenizer, tail)
        probe = _encode(self.tokenizer, self._word * 4 + tail + self._word * 4)
        expected = self._filler_ids * 4 + tail_ids + self._filler_ids * 4
        result = tail_ids if probe == expected else None
        self._tail_ids_cache[tail] = result
        return result

    def _fast_case_prompts(
        self, case_id: Any, total_len: int, cache_len: int, run_count: int
    ) -> CasePrompts:
        if cache_len == 0:
            texts, ids_list, specs = [], [], []
            for run_index in range(run_count):
                marker = f"case_{case_id}_cold_run_{run_index}_"
                marker_ids = self._safe_marker_ids(marker)
                if marker_ids is None or len(marker_ids) > total_len:
                    raise _FastConstructionUnavailable()
                fillers = total_len - len(marker_ids)
                texts.append(marker + self._word * fillers)
                ids_list.append(marker_ids + self._filler_ids * fillers)
                specs.append({"marker": marker, "fillers": fillers})
            return CasePrompts("", [], texts, ids_list, total_len, specs)
        marker = f"{self.SHARED_PREFIX_MARKER}{case_id}_"
        marker_ids = self._safe_marker_ids(marker)
        if marker_ids is None or len(marker_ids) > cache_len:
            raise _FastConstructionUnavailable()
        prefix_fillers = cache_len - len(marker_ids)
        prefix_ids = marker_ids + self._filler_ids * prefix_fillers
        seed_text = marker + self._word * prefix_fillers
        texts, ids_list, specs = [], [], []
        for run_index in range(run_count):
            tail = f" __run_{run_index}_"
            tail_ids = self._safe_tail_ids(tail)
            if tail_ids is None:
                raise _FastConstructionUnavailable()
            fillers = total_len - cache_len - len(tail_ids)
            if fillers < 0:
                raise _FastConstructionUnavailable()
            texts.append(seed_text + tail + self._word * fillers)
            ids_list.append(prefix_ids + tail_ids + self._filler_ids * fillers)
            specs.append({"tail": tail, "fillers": fillers})
        return CasePrompts(seed_text, prefix_ids, texts, ids_list, total_len, specs)

    def _verify(self, shape: str, prompts: CasePrompts) -> None:
        texts = ([prompts.seed_text] if prompts.seed_text else []) + prompts.run_texts
        ids_list = ([prompts.prefix_ids] if prompts.seed_text else []) + prompts.run_ids
        for text, ids in zip(texts, ids_list):
            if _encode(self.tokenizer, text) != ids:
                self._unsafe_shapes.add(shape)
                raise _FastConstructionUnavailable()
        self._verified_shapes.add(shape)

    def _text_case_prompts(
        self, case_id: Any, total_len: int, cache_len: int, run_count: int
    ) -> CasePrompts:
        _, prefix, built_len = self.make_case(case_id, total_len, cache_len)
        prefix_ids = _encode(self.tokenizer, prefix) if cache_len else []
        texts, ids_list = [], []
        for run_index in range(run_count):
            marker = (
                prefix + f" __run_{run_index}_"
                if cache_len
                else f"case_{case_id}_cold_run_{run_index}_"
            )
            text, ids = self._exact_text_and_ids(total_len, marker)
            if cache_len and ids[:cache_len] != prefix_ids:
                raise ValueError(f"run {run_index} did not preserve cache prefix")
            texts.append(text)
            ids_list.append(ids)
        return CasePrompts(prefix, prefix_ids, texts, ids_list, built_len, None)

    def build_case_prompts(
        self, case_id: Any, total_len: int, cache_len: int, run_count: int
    ) -> CasePrompts:
        if cache_len < 0 or cache_len >= total_len:
            raise ValueError(
                "cache_len must satisfy 0 <= cache_len < total_len, "
                f"got {cache_len}/{total_len}"
            )
        if run_count <= 0:
            raise ValueError(f"run_count must be positive, got {run_count}")
        shape = "cached" if cache_len else "cold"
        if shape not in self._unsafe_shapes:
            try:
                prompts = self._fast_case_prompts(
                    case_id, total_len, cache_len, run_count
                )
                if shape not in self._verified_shapes:
                    self._verify(shape, prompts)
                return prompts
            except _FastConstructionUnavailable:
                pass
        return self._text_case_prompts(case_id, total_len, cache_len, run_count)


def _normalize_grouped_case(case: Dict[str, Any]) -> Dict[str, Any]:
    groups = case.get("request_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("request_groups must be a non-empty array")
    normalized = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"request_groups[{index}] must be an object")
        count = group.get("count")
        input_len = group.get("input_len")
        cache_len = group.get("cache_len")
        if type(count) is not int or count <= 0:
            raise ValueError(f"request_groups[{index}].count must be positive")
        if type(input_len) is not int or input_len <= 0:
            raise ValueError(f"request_groups[{index}].input_len must be positive")
        if type(cache_len) is not int or not 0 <= cache_len < input_len:
            raise ValueError(f"request_groups[{index}].cache_len is invalid")
        normalized.append(
            {"count": count, "input_len": input_len, "cache_len": cache_len}
        )
    result = dict(case)
    result["request_groups"] = normalized
    expected_batch = sum(group["count"] for group in normalized)
    if result.get("batch_size") != expected_batch:
        raise ValueError("grouped summary batch_size must equal group count sum")
    for key in ("input_len", "cache_len"):
        if result.get(key) != max(group[key] for group in normalized):
            raise ValueError(f"grouped summary {key} must equal group maximum")
    if result.get("prefix_policy") not in {"shared_by_group", "independent"}:
        raise ValueError("invalid grouped prefix_policy")
    return result


def build_grouped_prompts(
    factory: PrefixPromptFactory, case: Dict[str, Any], run_count: int
):
    case = _normalize_grouped_case(case)
    members = []
    for group_index, group in enumerate(case["request_groups"]):
        shared = case["prefix_policy"] == "shared_by_group"
        shared_prompts = None
        for member in range(group["count"]):
            identity = f"{case['case_id']}_g{group_index}_r{0 if shared else member}"
            if shared_prompts is None:
                prompts = factory.build_case_prompts(
                    identity,
                    group["input_len"],
                    group["cache_len"],
                    run_count * group["count"] if shared else run_count,
                )
                if shared:
                    shared_prompts = prompts
            else:
                prompts = shared_prompts
            indices = (
                [run * group["count"] + member for run in range(run_count)]
                if shared
                else list(range(run_count))
            )
            members.append(
                {
                    "group_index": group_index,
                    "request_index": len(members),
                    "prefix_id": identity,
                    "input_len": group["input_len"],
                    "cache_len": group["cache_len"],
                    "prompts": CasePrompts(
                        prompts.seed_text,
                        prompts.prefix_ids,
                        [prompts.run_texts[index] for index in indices],
                        [prompts.run_ids[index] for index in indices],
                        prompts.built_len,
                    ),
                }
            )
    return members
