package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.util.List;

@Getter
@Setter
@ToString
@JsonIgnoreProperties(ignoreUnknown = true)
public class Request {
    @ToString.Exclude
    @JsonProperty("block_cache_keys")
    private List<Long> blockCacheKeys;

    @JsonProperty("seq_len")
    private long seqLen;

    @JsonProperty("request_id")
    private long requestId;

    @JsonProperty("generate_timeout")
    private long generateTimeout = 3600 * 1000;

    @JsonProperty("request_time_ms")
    private long requestTimeMs;

    @JsonProperty("api_key")
    @JsonAlias({"apikey", "apiKey"})
    @ToString.Exclude
    private String apiKey;

    // ============== V1-α DP Batching gating fields ==============
    // Mirror the pd_separation precondition (max_new_tokens > 1 && num_beams <= 1
    // && !force_disable_sp_run). RouteService inspects these to decide whether to
    // dispatch through DpBatchScheduler. Defaults intentionally fall back to the
    // legacy direct-route path so callers that don't populate them are safe.

    @JsonProperty("max_new_tokens")
    private int maxNewTokens = 1;

    @JsonProperty("num_beams")
    private int numBeams = 1;

    @JsonProperty("force_disable_sp_run")
    private boolean forceDisableSpRun = false;

    /**
     * Optional model name. Frontend passes this through MasterRequest payload so
     * DpBatchScheduler can shard PrefillQueues per model. Empty string falls back
     * to the global model (current behaviour).
     */
    @JsonProperty("model")
    private String model = "";

    /**
     * Base64-encoded protobuf bytes of the full {@code GenerateInputPB} built by
     * the frontend's {@code trans_input}. When present, {@code DpBatchScheduler.buildPb}
     * uses {@code GenerateInputPB.parseFrom(bytes).toBuilder()} as the basis and only
     * overlays {@code dp_rank} + {@code cache_hash_key} before sending {@code BatchEnqueue}
     * to Prefill. When empty (legacy/V0 callers), {@code buildPb} falls back to the
     * bare-PB construction.
     */
    @ToString.Exclude
    @JsonProperty("generate_input_pb_b64")
    private String generateInputPbB64 = "";
}
