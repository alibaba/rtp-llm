package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;
import lombok.extern.jackson.Jacksonized;

import java.util.List;
import java.util.concurrent.TimeUnit;

@Getter
@Builder(toBuilder = true)
@Jacksonized
@ToString
@JsonIgnoreProperties(ignoreUnknown = true)
public final class Request {
    public static final long DEFAULT_GENERATE_TIMEOUT_MS =
            TimeUnit.HOURS.toMillis(1L);

    @ToString.Exclude
    @JsonProperty("block_cache_keys")
    private final List<Long> blockCacheKeys;

    @ToString.Exclude
    @JsonProperty("cache_affinity_keys")
    @JsonAlias("media_keys")
    private final List<String> cacheAffinityKeys;

    @JsonProperty("selected_vit")
    private final ServerStatus selectedVit;

    @JsonProperty("vit_route_only")
    private final boolean vitRouteOnly;

    @JsonProperty("seq_len")
    private final long seqLen;

    @JsonProperty("cache_key_block_size")
    private final long cacheKeyBlockSize;

    @JsonProperty("request_id")
    private final long requestId;

    /** Upstream generation timeout retained for transport compatibility. */
    @JsonProperty("generate_timeout")
    @Builder.Default
    private final long generateTimeout = DEFAULT_GENERATE_TIMEOUT_MS;

    @JsonProperty("request_time_ms")
    private final long requestTimeMs;

    @JsonProperty("api_key")
    @JsonAlias({"apikey", "apiKey"})
    @ToString.Exclude
    private final String apiKey;

    @JsonProperty("max_new_tokens")
    @Builder.Default
    private final int maxNewTokens = 1;

    @JsonProperty("num_beams")
    @Builder.Default
    private final int numBeams = 1;

    @JsonProperty("force_disable_sp_run")
    @Builder.Default
    private final boolean forceDisableSpRun = false;


    @JsonProperty("model")
    @Builder.Default
    private final String model = "";

    /** Caller-supplied wire value; SchedulingMetadata owns the normalized priority. */
    @JsonProperty("priority")
    @Builder.Default
    private final int rawPriority = 0;

    private Request(List<Long> blockCacheKeys, List<String> cacheAffinityKeys,
                    ServerStatus selectedVit, boolean vitRouteOnly, long seqLen, long cacheKeyBlockSize,
                    long requestId, long generateTimeout, long requestTimeMs, String apiKey,
                    int maxNewTokens, int numBeams, boolean forceDisableSpRun, String model,
                    int rawPriority) {
        this.blockCacheKeys = blockCacheKeys == null ? List.of() : List.copyOf(blockCacheKeys);
        this.cacheAffinityKeys = cacheAffinityKeys == null ? List.of() : List.copyOf(cacheAffinityKeys);
        this.selectedVit = ServerStatus.copyOf(selectedVit);
        this.vitRouteOnly = vitRouteOnly;
        this.seqLen = seqLen;
        this.cacheKeyBlockSize = cacheKeyBlockSize;
        this.requestId = requestId;
        this.generateTimeout = generateTimeout;
        this.requestTimeMs = requestTimeMs;
        this.apiKey = apiKey;
        this.maxNewTokens = maxNewTokens;
        this.numBeams = numBeams;
        this.forceDisableSpRun = forceDisableSpRun;
        this.model = model;
        this.rawPriority = rawPriority;
    }

    public ServerStatus getSelectedVit() {
        return ServerStatus.copyOf(selectedVit);
    }

    @JsonIgnore
    public long getHardKvTokens() {
        return Math.max(0L, seqLen);
    }

    @JsonIgnore
    public long getExpectedKvTokens() {
        long prompt = getHardKvTokens();
        long output = Math.max(0L, maxNewTokens);
        return prompt > Long.MAX_VALUE - output ? Long.MAX_VALUE : prompt + output;
    }
}
