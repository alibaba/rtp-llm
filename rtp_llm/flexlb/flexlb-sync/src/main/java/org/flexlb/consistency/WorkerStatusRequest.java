package org.flexlb.consistency;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * @author zjw
 * description:
 * date: 2025/3/31
 */
@Setter
@Getter
@AllArgsConstructor
@NoArgsConstructor

public class WorkerStatusRequest {
    @JsonProperty("latest_cache_version")
    private long latestCacheVersion;

    @JsonProperty("latest_finised_version")
    private long latestFinisedVersion;

}
