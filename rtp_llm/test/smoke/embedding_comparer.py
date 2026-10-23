import json
import logging
import os
from typing import Any, Dict, List

import torch
from pydantic import BaseModel
from smoke.base_comparer import BaseComparer
from smoke.common_def import ABS_PATH, REL_PATH, QueryStatus, SmokeException
from smoke.utils import create_temporary_copy, save_hidden_states

from rtp_llm.models.downstream_modules.embedding.api_datatype import (
    AllEmbeddingRequest,
    ALLEmbeddingResponse,
    ChatMessage,
    ContentPart,
    EmbeddingResponseType,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
    SparseEmbeddingRequest,
)

EXPECT_EMBEDDING_PATH_KEY = "embedding_path"


class EmbeddingComparer(BaseComparer):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.rtol = 1e-2
        self.atol = 1e-2

    def format_query(self, query_json: Dict[str, Any]) -> BaseModel:
        if "return_decoded" in query_json:
            query_info = SparseEmbeddingRequest(**query_json)
        elif "normalize" in query_json:
            query_info = AllEmbeddingRequest(**query_json)
        else:
            query_info = OpenAIEmbeddingRequest(**query_json)
            self._rewrite_query_info(query_info)
        return query_info

    def _maybe_load_from_path(self, res: Any) -> Any:
        if isinstance(res, str) and os.path.exists(os.path.join(REL_PATH, res)):
            return torch.load(os.path.join(REL_PATH, res), weights_only=False)
        return res

    def format_result(self, result_json: Dict[str, Any]) -> BaseModel:
        logging.info("result_json: %s", result_json)
        if "token_ids" in result_json["data"][0]:
            res = ALLEmbeddingResponse(**result_json)
        else:
            res = OpenAIEmbeddingResponse(**result_json)
        if save_hidden_states():
            return res
        for single_res in res.data:
            single_res.embedding = self._maybe_load_from_path(single_res.embedding)
        return res

    def curl_response_to_json(
        self, query_info: Any, curl_response: Any
    ) -> Dict[str, Any]:
        return json.loads(curl_response)

    def _compare_list_result(self, expect: List[float], actual: List[float]):
        res = torch.isclose(
            torch.tensor(expect), torch.tensor(actual), rtol=self.rtol, atol=self.atol
        ).reshape(-1)
        if not all(res):
            raise SmokeException(QueryStatus.COMPARE_FAILED, f"list data at not equal")

    def _compare_dict_result(self, expect: Dict[str, float], actual: Dict[str, float]):
        for key in expect.keys():
            if key not in actual:
                raise Exception(f"failed to find {key} in actual response")
            res = torch.isclose(
                torch.tensor(expect[key]),
                torch.tensor(actual[key]),
                rtol=self.rtol,
                atol=self.atol,
            ).reshape(-1)
            if not all(res):
                raise SmokeException(
                    QueryStatus.COMPARE_FAILED, f"data at key: {key} not equal"
                )

    def compare_result(
        self,
        expect_result: OpenAIEmbeddingResponse,
        actual_result: OpenAIEmbeddingResponse,
    ) -> None:
        if type(expect_result) != type(actual_result):
            raise SmokeException(QueryStatus.COMPARE_FAILED, f"type not equal")
        if len(expect_result.data) != len(actual_result.data):
            raise SmokeException(QueryStatus.COMPARE_FAILED, f"len not equal")
        for expect, actual in zip(expect_result.data, actual_result.data):
            if expect_result.object != actual_result.object:
                raise SmokeException(
                    QueryStatus.COMPARE_FAILED, f"embedding object type not equal"
                )
            if expect.object == EmbeddingResponseType.DENSE:
                self._compare_list_result(expect.embedding, actual.embedding)
            elif expect.object == EmbeddingResponseType.SPARSE:
                self._compare_dict_result(expect.embedding, actual.embedding)
            elif expect.object == EmbeddingResponseType.COLBERT:
                self._compare_list_result(expect.embedding, actual.embedding)
            else:
                raise SmokeException(
                    QueryStatus.COMPARE_FAILED,
                    f"unknown embedding object type: {expect.object}",
                )

            if expect.index != actual.index:
                raise SmokeException(
                    QueryStatus.COMPARE_FAILED,
                    f"index not equal expect.index = {expect.index}, actual.index = {actual.index}",
                )

            if hasattr(expect, "token_ids") and expect.token_ids != actual.token_ids:
                raise SmokeException(
                    QueryStatus.COMPARE_FAILED,
                    f"token_ids not equal expect.token_ids = {expect.token_ids}, actual.token_ids = {actual.token_ids}",
                )

    def _maybe_rewrite_expect_result(
        self,
        actual: OpenAIEmbeddingResponse,
        expect: OpenAIEmbeddingResponse,
        query_info: OpenAIEmbeddingRequest,
    ) -> None:
        if not save_hidden_states():
            return

        assert len(actual.data) == len(
            expect.data
        ), "expect response data len not equals to actual response data len"
        out_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())
        rewrite_dir = os.path.join(out_dir, "smoke_actual")
        os.makedirs(rewrite_dir, exist_ok=True)
        for index in range(len(actual.data)):
            if isinstance(expect.data[index].embedding, str):
                pt_name = os.path.basename(expect.data[index].embedding)
                torch.save(actual.data[index].embedding, os.path.join(rewrite_dir, pt_name))

    def _rewrite_query_info(self, query_info: OpenAIEmbeddingRequest):
        def _rewrite_content_part(part: ContentPart):
            if part.image_url is not None:
                part.image_url.url = create_temporary_copy(part.image_url.url)
            if part.video_url is not None:
                part.video_url.url = create_temporary_copy(part.video_url.url)

        if isinstance(query_info.input, ContentPart):
            _rewrite_content_part(query_info.input)
        elif isinstance(query_info.input, ChatMessage) and isinstance(
            query_info.input.content, List
        ):
            for part in query_info.input.content:
                _rewrite_content_part(part)
        elif isinstance(query_info.input, list) and len(query_info.input) > 0:
            if isinstance(query_info.input[0], ContentPart):
                for part in query_info.input:
                    _rewrite_content_part(part)
            elif isinstance(query_info.input[0], ChatMessage):
                for message in query_info.input:
                    if isinstance(message.content, List):
                        for part in message.content:
                            _rewrite_content_part(part)

        return
