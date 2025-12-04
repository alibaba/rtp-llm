"""
Pipeline Response Data Models

包含所有与Pipeline响应相关的数据模型定义
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class PipelineResponse(BaseModel):
    """单个Pipeline响应的数据模型"""

    response: str = ""
    finished: bool = True
    aux_info: Dict[str, Any] = {}
    hidden_states: Optional[Union[List[float], List[List[float]]]] = None
    loss: Optional[Union[float, List[float]]] = None
    logits: Optional[Union[List[float], List[List[float]]]] = None
    output_ids: Optional[List[List[int]]] = None
    input_ids: Optional[List[List[int]]] = None


class MultiSequencesPipelineResponse(BaseModel):
    """多序列Pipeline响应的数据模型"""

    response: List[str]
    finished: bool
    aux_info: List[Dict[str, Any]] = {}


class BatchPipelineResponse(BaseModel):
    """批处理Pipeline响应的数据模型"""

    response_batch: List[Union[PipelineResponse, MultiSequencesPipelineResponse]]


class TokenizerEncodeResponse(BaseModel):
    """Tokenizer编码响应的数据模型"""

    token_ids: List[int] = []
    offset_mapping: Optional[List[Any]] = None
    tokens: List[str] = []
    error: str = ""
