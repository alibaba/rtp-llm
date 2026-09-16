"""Test-only acknowledgement for copied DashSc traffic; never consumes output."""

import logging
import os


async def acknowledge(visitor, generate_input, request):
    from rtp_llm.dash_sc.proto import predict_v2_pb2

    visitor.fill_request_info(generate_input)
    generate_input.generate_config.validate()
    if generate_input.prompt_length <= 0:
        raise ValueError("empty schedule-only input")
    await visitor.route_ips(generate_input)
    completed = False
    if not generate_input.enqueued_by_master:
        if os.environ.get("RTP_LLM_MOCK_NON_BATCH") != "1":
            raise ValueError("NON_BATCH requires RTP_LLM_MOCK_NON_BATCH=1")
        # The existing RPC client selects GenerateStreamCall when master did not
        # enqueue. Consume its terminal result; never cancel after a first frame
        # and never call FetchResponse. Errors/cancellation propagate normally.
        async for result in visitor.model_rpc_client.enqueue(generate_input):
            if result.generate_outputs:
                completed = all(output.finished for output in result.generate_outputs)
        if not completed:
            raise RuntimeError("mock NON_BATCH stream ended without a terminal result")
    response = predict_v2_pb2.ModelStreamInferResponse()
    infer = response.infer_response
    infer.id = request.id
    infer.model_name = request.model_name
    infer.parameters["mock_schedule_only"].bool_param = True
    infer.parameters["schedule_accepted"].bool_param = True
    infer.parameters["inference_completed"].bool_param = completed
    # No generated tokens, finished=true, or fabricated inference success.
    logging.info("mock_schedule_accepted request_id=%s inference_completed=%s",
                 generate_input.request_id, completed)
    return response
