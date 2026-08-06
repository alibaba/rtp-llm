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

    @JsonProperty("cache_key_block_size")
    private long cacheKeyBlockSize;

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

    @JsonProperty("max_new_tokens")
    private int maxNewTokens = 1;

    @JsonProperty("num_beams")
    private int numBeams = 1;

    @JsonProperty("force_disable_sp_run")
    private boolean forceDisableSpRun = false;

    @JsonProperty("model")
    private String model = "";

    /**
     * Auto-TPM QoS priority. Valid values: 30/40/50/60/70; 0 means "no
     * priority" (request opts out of Auto-TPM and is scheduled on the legacy
     * path). Explicit invalid values are normalized to 50 upstream. See
     * {@code PriorityNormalizer}.
     */
    @JsonProperty("priority")
    private int priority = 0;

    /** True iff the request carries an explicit Auto-TPM priority. */
    public boolean hasPriority() {
        return priority > 0;
    }
}
