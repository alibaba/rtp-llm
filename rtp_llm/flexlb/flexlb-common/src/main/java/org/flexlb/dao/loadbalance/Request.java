package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.util.List;

@Getter
@Setter
@ToString
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
}
