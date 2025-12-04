"""
Response Formatter

处理响应格式化逻辑，将生成的结果转换为标准响应格式
"""

from dataclasses import asdict
from typing import Any, Dict

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.utils.base_model_datatypes import GenerateResponse

from .pipeline_response import MultiSequencesPipelineResponse, PipelineResponse


class ResponseFormatter:
    """响应格式化器，负责将生成结果格式化为标准响应"""

    def format_response(
        self, gen_responses: GenerateResponse, generate_config: GenerateConfig
    ) -> Dict[str, Any]:
        """格式化单个响应"""
        generate_texts = gen_responses.generate_texts
        finished = gen_responses.generate_outputs.generate_outputs[0].finished
        aux_info = None
        if generate_config.aux_info:
            aux_info = gen_responses.generate_outputs.generate_outputs[0].aux_info
            if generate_config.has_num_beams():
                aux_info.beam_responses = generate_texts

        hidden_states = gen_responses.generate_outputs.generate_outputs[0].hidden_states
        output_ids = gen_responses.generate_outputs.generate_outputs[0].output_ids
        input_ids = gen_responses.generate_outputs.generate_outputs[0].input_ids
        loss = gen_responses.generate_outputs.generate_outputs[0].loss
        logits = gen_responses.generate_outputs.generate_outputs[0].logits

        response = PipelineResponse(
            response=generate_texts[0],
            finished=finished,
            aux_info=asdict(aux_info) if generate_config.aux_info else {},
            hidden_states=(
                hidden_states.tolist()
                if generate_config.return_hidden_states and hidden_states is not None
                else None
            ),
            loss=(
                loss.tolist()
                if generate_config.calculate_loss and loss is not None
                else None
            ),
            logits=(
                logits.tolist()
                if generate_config.return_logits and logits is not None
                else None
            ),
            output_ids=(
                output_ids.tolist()
                if generate_config.return_output_ids and output_ids is not None
                else None
            ),
            input_ids=(
                input_ids.tolist()
                if generate_config.return_input_ids and input_ids is not None
                else None
            ),
        )
        return response

    def format_response_new(
        self, gen_responses: GenerateResponse, generate_config: GenerateConfig
    ) -> Dict[str, Any]:
        """格式化新版响应（支持多序列）"""
        generate_texts = gen_responses.generate_texts
        if generate_config.num_return_sequences > 0:
            aux_info = []
            if generate_config.aux_info:
                aux_info = [
                    asdict(seq.aux_info)
                    for seq in gen_responses.generate_outputs.generate_outputs
                ]
            sequences_pipeline_response = MultiSequencesPipelineResponse(
                response=generate_texts,
                finished=all(
                    [
                        seq.finished
                        for seq in gen_responses.generate_outputs.generate_outputs
                    ]
                ),
                aux_info=aux_info,
            )
            return sequences_pipeline_response
        else:
            return self.format_response(gen_responses, generate_config)
