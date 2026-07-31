#pragma once

namespace rtp_llm {

// Single receive-size cap shared by the model-rpc server (RtpLLMOp) and the
// client channel pool (RPCPool), so the two defaults cannot drift apart into
// a "client can send it, server rejects it" split.
//
// Payload audit (2026-07, messages on model_rpc_service.proto): the largest
// request-direction tensors are the dspark PD side-channel propose probs
// ([1, k, vocab] fp32, ~4.3 MiB at k=7 / vocab=152k) and multimodal
// embeddings (tens of MiB for long vision inputs); response-direction
// all_probs/all_hidden_states scale with batch*vocab (hundreds of MiB at
// pathological batch sizes). Context KV transfers go through the cache
// store, not this gRPC channel. 1 GiB keeps comfortable headroom over all
// of these while staying finite, so one oversized message cannot force an
// unbounded allocation. An explicit server_config entry still overrides the
// server side.
inline constexpr int kGrpcMaxReceiveMessageBytes = 1 << 30;  // 1 GiB

}  // namespace rtp_llm
