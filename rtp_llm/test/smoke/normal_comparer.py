import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel, ValidationError
from smoke.base_comparer import BaseComparer
from smoke.common_def import ABS_PATH, REL_PATH, QueryStatus, SmokeException
from smoke.utils import create_temporary_copy, save_hidden_states, save_logits
from typing import Any, Callable, Optional

from rtp_llm.config.generate_config import GenerateConfig

EXPECT_HIDDEN_STATES_KEY = "expected_hidden_states_path"
EXPECT_LOGITS_KEY = "expected_logits_path"


# corresponding to query in q_r.json
class QueryInfo(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    # List[Any] for ChatAPI format
    prompt: Optional[Union[List[Any], str]] = None
    prompt_batch: Optional[Union[List[str], List[List[Any]]]] = None
    images: Optional[Union[List[List[str]], List[str]]] = None
    generate_config: GenerateConfig = GenerateConfig()
    yield_generator: bool = False

    def check_vaild(self):
        if self.prompt is None and self.prompt_batch is None:
            raise SmokeException(
                QueryStatus.VALID_FAILED,
                f"neither prompt and prompt_batch in request: {self.model_dump()}",
            )
        elif self.prompt is not None and self.prompt_batch is not None:
            raise SmokeException(
                QueryStatus.VALID_FAILED,
                f"both prompt and prompt_batch in request: {self.model_dump()}",
            )

    @property
    def is_batch(self):
        return self.prompt_batch is not None

class AuxInfo(BaseModel):
    input_len: Optional[int] = None
    prefix_len: Optional[int] = None
    reuse_len: Optional[int] = None
    local_reuse_len: Optional[int] = None
    remote_reuse_len: Optional[int] = None
    memory_reuse_len: Optional[int] = None
    prefill_total_reuse_len: Optional[int] = None
    prefill_local_reuse_len: Optional[int] = None
    prefill_remote_reuse_len: Optional[int] = None
    prefill_memory_reuse_len: Optional[int] = None
    decode_total_reuse_len: Optional[int] = None
    decode_local_reuse_len: Optional[int] = None
    decode_remote_reuse_len: Optional[int] = None
    decode_memory_reuse_len: Optional[int] = None

    output_len: Optional[int] = None
    step_output_len: Optional[int] = None
    iter_count: Optional[int] = None
    cum_log_probs: Optional[Union[List[float], List[None]]] = None
    beam_responses: Optional[List[str]] = None
    pd_sep: Optional[bool] = None
    softmax_probs: Optional[List[float]] = None


class SmokeResponse(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    response: Union[str, List[str]]
    response_alternatives: Optional[List[Union[str, List[str]]]] = None
    hidden_states: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    loss: Optional[Union[float, List[float]]] = None
    input_ids: Optional[List[List[int]]] = None
    output_ids: Optional[List[List[int]]] = None
    aux_info: Optional[Union[AuxInfo, List[AuxInfo]]] = None

    def __init__(
        self,
        hidden_states: Optional[Union[List[float], torch.Tensor]] = None,
        logits: Optional[Union[List[float], torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        def _conver_tensor(x: Optional[Union[List[float], torch.Tensor]]):
            if x is not None:
                return torch.tensor(x, dtype=torch.float)
            return x

        hidden_states = _conver_tensor(hidden_states)
        logits = _conver_tensor(logits)
        super().__init__(logits=logits, hidden_states=hidden_states, *args, **kwargs)


class SmokeReponseList(BaseModel):
    response_batch: List[SmokeResponse] = []


class NormalComparer(BaseComparer):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.test_logits = os.environ.get("TEST_LOGITS", "False") == "True"
        self.test_hidden_states = (
            os.environ.get("TEST_HIDDEN_STATES", "False") == "True"
        )
        self.logits_path = ""
        self.hidden_states_path = ""

    # override
    def curl_response_to_json(self, query_info: QueryInfo, res: Any) -> Dict[str, Any]:
        if query_info.yield_generator:
            res = list(filter(None, res))
            if query_info.generate_config.return_incremental:
                chunks = [json.loads(chunk.decode("utf-8")[5:]) for chunk in res[:-1]]
                res = {
                    "response": "".join(chunk["response"] for chunk in chunks),
                    "aux_info": chunks[-1]["aux_info"],
                    "finished": chunks[-1]["finished"],
                }
            else:
                res = json.loads(res[-2].decode("utf-8")[5:])  # data:
        else:
            res = json.loads(res)
        return res

    # override
    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        query_info = QueryInfo(**query_json)
        self._rewrite_query(query_info)
        return query_info
    
    def get_concurrency_batch(self, query_info: QueryInfo) -> int:
        if query_info.prompt_batch is not None:
            return len(query_info.prompt_batch)
        return 1

    # override
    def format_result(self, result_json: Dict[str, Any]) -> Any:
        hidden_states = None
        logits = None
        # 如果rewrite结果的话，不需要load tensor，同时文件路径需要是绝对路径而非bazel临时路径
        if EXPECT_HIDDEN_STATES_KEY in result_json:
            if "SAVE_HIDDEN_STATES" not in os.environ:
                self.hidden_states_path = os.path.join(
                    REL_PATH, result_json[EXPECT_HIDDEN_STATES_KEY]
                )
                hidden_states = torch.load(self.hidden_states_path, weights_only=False)
                del result_json[EXPECT_HIDDEN_STATES_KEY]
            else:
                out_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
                self.hidden_states_path = os.path.join(
                    out_dir, "smoke_actual", result_json[EXPECT_HIDDEN_STATES_KEY]
                )
        if EXPECT_LOGITS_KEY in result_json:
            if "SAVE_LOGITS" not in os.environ:
                self.logits_path = os.path.join(
                    REL_PATH, result_json[EXPECT_LOGITS_KEY]
                )
                logits = torch.load(self.logits_path, weights_only=False)
                del result_json[EXPECT_LOGITS_KEY]
            else:
                out_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
                self.logits_path = os.path.join(
                    out_dir, "smoke_actual", result_json[EXPECT_LOGITS_KEY]
                )

        try:
            if "response_batch" in result_json:
                smoke_response_list = SmokeReponseList()
                for idx, response in enumerate(result_json["response_batch"]):
                    res = SmokeResponse(**response)
                    if hidden_states is not None:
                        res.hidden_states = hidden_states[idx]
                    if logits is not None:
                        res.logits = logits[idx]
                    smoke_response_list.response_batch.append(res)
                return smoke_response_list
            else:
                res = SmokeResponse(**result_json)
                if hidden_states is not None:
                    res.hidden_states = hidden_states
                if logits is not None:
                    res.logits = logits
                return res
        except ValidationError as e:
            logging.info(f"result_json is not valid, result_json: {result_json}")
            raise e

    def _format_all_diffs(self, diffs: List[str]) -> str:
        """Format collected diffs into a single error message."""
        if not diffs:
            return ""
        n = len(diffs)
        lines = [
            "",
            "=" * 60,
            f"  Compare failed: {n} difference(s) found",
            "=" * 60,
        ]
        for i, d in enumerate(diffs, 1):
            lines.append("")
            lines.append(f"  --- Diff {i}/{n} ---")
            lines.append("")
            # Indent each line of this diff for readability
            for line in d.split("\n"):
                lines.append("    " + line if line.strip() else "")
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def compare_result(
        self,
        expect: Union[SmokeReponseList, SmokeResponse],
        actual: Union[SmokeReponseList, SmokeResponse],
    ):
        assert type(expect) == type(
            actual
        ), f"type different: expect:{expect} vs actual:{actual}"
        if isinstance(expect, SmokeReponseList):
            assert isinstance(actual, SmokeReponseList)
            if len(actual.response_batch) != len(expect.response_batch):
                raise SmokeException(
                    QueryStatus.VALID_FAILED,
                    f"len different {len(expect.response_batch)} vs {len(actual.response_batch)}:  expect:{expect}, actual:{actual})",
                )
            diffs: List[str] = []
            for idx in range(len(actual.response_batch)):
                self._compare(
                    expect.response_batch[idx],
                    actual.response_batch[idx],
                    diffs,
                    prefix=f"[batch_idx={idx}] ",
                )
            if diffs:
                raise SmokeException(
                    QueryStatus.COMPARE_FAILED,
                    self._format_all_diffs(diffs),
                )
        else:
            assert isinstance(actual, SmokeResponse)
            diffs = []
            self._compare(expect, actual, diffs)
            if diffs:
                raise SmokeException(
                    QueryStatus.COMPARE_FAILED,
                    self._format_all_diffs(diffs),
                )

    def _format_beam_responses_diff(
        self, expect_beams: Optional[List[str]], actual_beams: Optional[List[str]]
    ) -> str:
        """Format beam_responses diff for error message."""
        if expect_beams is None and actual_beams is None:
            return ""
        exp_len = len(expect_beams) if expect_beams else 0
        act_len = len(actual_beams) if actual_beams else 0
        lines = [
            "beam_responses:",
            "",
        ]
        if exp_len != act_len:
            lines.extend([
                "  length:",
                f"    expect: {exp_len}",
                f"    actual:  {act_len}",
                "",
            ])
        lines.append("  expect (full list):")
        for i, s in enumerate(expect_beams or []):
            lines.append(f"    [{i}] {repr(s)}")
        lines.append("")
        lines.append("  actual (full list):")
        for i, s in enumerate(actual_beams or []):
            lines.append(f"    [{i}] {repr(s)}")
        lines.append("")
        lines.append("  per-index diff:")
        max_len = max(exp_len, act_len)
        any_diff = False
        for i in range(max_len):
            exp = expect_beams[i] if expect_beams and i < len(expect_beams) else "<missing>"
            act = actual_beams[i] if actual_beams and i < len(actual_beams) else "<missing>"
            if exp != act:
                any_diff = True
                lines.append(f"    [{i}] expect: {repr(exp)}")
                lines.append(f"    [{i}] actual:  {repr(act)}")
                lines.append("")
        if not any_diff and exp_len == act_len:
            lines.append("    (none)")
        return "\n".join(lines).rstrip()

    def _compare_aux_info(
        self,
        expect_aux: AuxInfo,
        actual_aux: AuxInfo,
        diffs: List[str],
        prefix: str = "",
    ) -> None:
        """Compare aux_info fields and append any diffs to diffs list (no raise)."""

        def check_equal(
            field_name: str,
            expect_val: Any,
            actual_val: Any,
            comparator: Callable[[Any, Any], bool] = lambda x, y: x == y,
        ) -> None:
            if expect_val is not None and not comparator(expect_val, actual_val):
                msg = f"{prefix}aux_info.{field_name}:\n    expect: {repr(expect_val)}\n    actual:  {repr(actual_val)}"
                if field_name == "beam_responses":
                    msg += "\n\n" + self._format_beam_responses_diff(
                        getattr(expect_aux, "beam_responses", None),
                        getattr(actual_aux, "beam_responses", None),
                    )
                diffs.append(msg)

        # 普通字段直接比较
        for field in [
            "input_len", "prefix_len", "reuse_len", "output_len", "iter_count",
            "local_reuse_len", "remote_reuse_len", "memory_reuse_len",
            "prefill_total_reuse_len", "prefill_local_reuse_len",
            "prefill_remote_reuse_len", "prefill_memory_reuse_len",
            "decode_total_reuse_len", "decode_local_reuse_len",
            "decode_remote_reuse_len", "decode_memory_reuse_len",
        ]:
            expect_val = getattr(expect_aux, field)
            actual_val = getattr(actual_aux, field)
            check_equal(field, expect_val, actual_val)

        check_equal("beam_responses", expect_aux.beam_responses, actual_aux.beam_responses)

        def is_close_list(a: Any, b: Any) -> bool:
            if a is None or b is None:
                return a == b
            if len(a) != len(b):
                return False
            return bool(torch.all(torch.isclose(
                torch.tensor(a), torch.tensor(b), rtol=1e-2, atol=1e-2
            )))

        check_equal(
            "softmax_probs",
            expect_aux.softmax_probs,
            actual_aux.softmax_probs,
            is_close_list,
        )
        check_equal(
            "cum_log_probs",
            expect_aux.cum_log_probs,
            actual_aux.cum_log_probs,
            is_close_list,
        )

    def _compare(
        self,
        expect: SmokeResponse,
        actual: SmokeResponse,
        diffs: List[str],
        prefix: str = "",
    ) -> None:
        """Compare expect vs actual in full; append all diffs to diffs (no raise)."""
        rtol = atol = 1e-2

        # response
        if expect.response != actual.response:
            if expect.response_alternatives and actual.response in expect.response_alternatives:
                logging.info(
                    f"[STABILITY_DIAG] Response matched alternative: "
                    f"primary=[{expect.response}] actual=[{actual.response}] "
                    f"alternatives={expect.response_alternatives}"
                )
            else:
                msg = (
                    f"{prefix}response:\n"
                    f"  expect: {repr(expect.response)}\n"
                    f"  actual:  {repr(actual.response)}"
                )
                if expect.aux_info is not None and actual.aux_info is not None:
                    exp_beams = getattr(expect.aux_info, "beam_responses", None)
                    act_beams = getattr(actual.aux_info, "beam_responses", None)
                    if exp_beams is not None or act_beams is not None:
                        msg += "\n\n" + self._format_beam_responses_diff(exp_beams, act_beams)
                diffs.append(msg)

        # loss
        if expect.loss is not None:
            loss_cmp = torch.isclose(
                torch.tensor(expect.loss),
                torch.tensor(actual.loss),
                rtol=rtol,
                atol=atol,
            )
            if not all(loss_cmp.reshape(-1)):
                diffs.append(
                    f"{prefix}loss:\n    expect: {expect.loss}\n    actual:  {actual.loss}"
                )

        # hidden_states
        if self.test_hidden_states and expect.hidden_states is not None:
            if actual.hidden_states is None:
                diffs.append(f"{prefix}hidden_states:\n    actual: None (missing)")
            elif not all(
                torch.isclose(
                    expect.hidden_states,
                    actual.hidden_states,
                    rtol=rtol,
                    atol=atol,
                ).reshape(-1)
            ):
                diffs.append(
                    f"{prefix}hidden_states (not close):\n"
                    f"    expect shape: {expect.hidden_states.shape}\n"
                    f"    actual shape:  {actual.hidden_states.shape}"
                )

        # logits — only verify shape; values are non-deterministic across GPU runs
        if self.test_logits and expect.logits is not None:
            if actual.logits is None:
                diffs.append(f"{prefix}logits:\n    actual: None (missing)")
            elif expect.logits.shape != actual.logits.shape:
                diffs.append(
                    f"{prefix}logits shape mismatch:\n"
                    f"    expect shape: {expect.logits.shape}\n"
                    f"    actual shape:  {actual.logits.shape}"
                )

        # output_ids
        if expect.output_ids is not None and actual.output_ids != expect.output_ids:
            diffs.append(
                f"{prefix}output_ids:\n    expect: {expect.output_ids}\n    actual:  {actual.output_ids}"
            )

        # input_ids
        if expect.input_ids is not None and actual.input_ids != expect.input_ids:
            diffs.append(
                f"{prefix}input_ids:\n    expect: {expect.input_ids}\n    actual:  {actual.input_ids}"
            )

        # aux_info: skip comparison when expected auxinfo is null
        if expect.aux_info is not None and actual.aux_info is not None:
            self._compare_aux_info(
                expect.aux_info, actual.aux_info, diffs, prefix=prefix
            )

    def _rewrite_images(self, images: Union[List[str], str]) -> Union[List[str], str]:
        # iter rewrite
        if isinstance(images, list):
            return [self._rewrite_images(x) for x in images]
        return create_temporary_copy(images)

    def _rewrite_query(self, query_info: QueryInfo) -> None:
        if isinstance(query_info.prompt, str) and os.path.isfile(query_info.prompt):
            with open(query_info.prompt, "r") as f:
                content = f.read()
                query_info.prompt = content

        if query_info.generate_config.return_logits:
            query_info.generate_config.return_logits = self.test_logits

        if query_info.generate_config.return_hidden_states:
            query_info.generate_config.return_hidden_states = self.test_hidden_states

        if query_info.images is not None:
            query_info.images = self._rewrite_images(query_info.images)

    def _maybe_rewrite_expect_result(
        self,
        smoke_response: Union[SmokeReponseList, SmokeResponse],
        expect_response: Union[SmokeReponseList, SmokeResponse],
        query_info: QueryInfo,
    ):
        if (
            self.logits_path != ""
            and save_logits()
            and query_info.generate_config.return_logits
        ):

            if isinstance(smoke_response, SmokeReponseList):
                logits = [x.logits for x in smoke_response.response_batch]
                logits = torch.stack(logits)
            else:
                logits = smoke_response.logits
            os.makedirs(os.path.dirname(self.logits_path), exist_ok=True)
            torch.save(torch.tensor(logits, dtype=torch.float), self.logits_path)

        if (
            self.hidden_states_path != ""
            and save_hidden_states()
            and query_info.generate_config.return_hidden_states
        ):

            if isinstance(smoke_response, SmokeReponseList):
                hidden_states = [x.hidden_states for x in smoke_response.response_batch]
                hidden_states = torch.stack(hidden_states)
            else:
                hidden_states = smoke_response.hidden_states
            os.makedirs(os.path.dirname(self.hidden_states_path), exist_ok=True)
            torch.save(
                torch.tensor(hidden_states, dtype=torch.float), self.hidden_states_path
            )
