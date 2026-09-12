"""Test-only acknowledgement for copied DashSc traffic; never consumes output."""

import logging


async def acknowledge(visitor, generate_input, request):
    from rtp_llm.dash_sc.proto import predict_v2_pb2

    visitor.fill_request_info(generate_input)
    generate_input.generate_config.validate()
    if generate_input.prompt_length <= 0:
        raise ValueError("empty schedule-only input")
    await visitor.route_ips(generate_input)
    if not generate_input.enqueued_by_master:
        raise ValueError("schedule-only requires master BATCH enqueue")
    response = predict_v2_pb2.ModelStreamInferResponse()
    infer = response.infer_response
    infer.id = request.id
    infer.model_name = request.model_name
    infer.parameters["mock_schedule_only"].bool_param = True
    infer.parameters["schedule_accepted"].bool_param = True
    infer.parameters["inference_completed"].bool_param = False
    # No generated tokens, finished=true, or fabricated inference success.
    logging.info("mock_schedule_accepted request_id=%s inference_completed=false",
                 generate_input.request_id)
    return response
