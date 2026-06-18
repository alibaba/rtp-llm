package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.route.RoleType;

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

    @JsonProperty("excluded_workers")
    private List<ExcludedWorker> excludedWorkers;

    public boolean isWorkerExcluded(RoleType roleType, String ip, int port) {
        if (excludedWorkers == null || excludedWorkers.isEmpty() || StringUtils.isBlank(ip)) {
            return false;
        }
        for (ExcludedWorker excludedWorker : excludedWorkers) {
            if (excludedWorker == null || excludedWorker.getRole() != roleType) {
                continue;
            }
            if (!ip.equals(excludedWorker.getServerIp())) {
                continue;
            }
            int excludedPort = excludedWorker.getHttpPort();
            if (excludedPort <= 0 || excludedPort == port) {
                return true;
            }
        }
        return false;
    }
}
