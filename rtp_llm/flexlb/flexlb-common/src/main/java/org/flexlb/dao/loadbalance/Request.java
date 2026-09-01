package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
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
    public static final long DEFAULT_GENERATE_TIMEOUT_MS = 60L * 60L * 1000L;
    public static final int SESSION_SCHEMA_VERSION = 1;

    @ToString.Exclude
    @JsonProperty("block_cache_keys")
    private List<Long> blockCacheKeys;

    @JsonProperty("seq_len")
    private long seqLen;

    @JsonProperty("cache_key_block_size")
    private long cacheKeyBlockSize;

    @JsonProperty("request_id")
    private long requestId;

    /** Upstream generation timeout retained for transport compatibility. */
    @JsonProperty("generate_timeout")
    private long generateTimeout = DEFAULT_GENERATE_TIMEOUT_MS;

    @JsonProperty("request_time_ms")
    private long requestTimeMs;

    @JsonProperty("api_key")
    @JsonAlias({"apikey", "apiKey"})
    @ToString.Exclude
    private String apiKey;

    @JsonProperty("max_new_tokens")
    private int maxNewTokens = 1;

    @JsonProperty("num_beams")
    private int numBeams = 1;

    @JsonProperty("force_disable_sp_run")
    private boolean forceDisableSpRun = false;

    @JsonProperty("model")
    private String model = "";

    /**
     * priority scheduling QoS priority. Valid range: 1-100 (higher = more important).
     * Set by {@code PriorityNormalizer.normalize()} upstream — always 1-100
     * in production; the proto3 default 0 only appears before normalization.
     * See {@code PriorityNormalizer}.
     */
    @JsonProperty("priority")
    private int priority = 0;

    private int sessionSchemaVersion;

    private String inferenceSessionId = "";

    private SessionState inferenceSessionState = SessionState.UNSPECIFIED;

    @JsonIgnore
    private long sessionPlacementEpoch = -1L;

    public enum SessionState {
        UNSPECIFIED,
        NEW,
        ESTABLISHED
    }

}
