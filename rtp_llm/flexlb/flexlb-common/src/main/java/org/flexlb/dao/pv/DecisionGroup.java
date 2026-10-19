package org.flexlb.dao.pv;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

/** One committed scheduling group, independent of engine batch IDs and later delivery outcomes. */
public record DecisionGroup(String id,
                            String policy,
                            String dispatcher,
                            String worker,
                            int committedSize,
                            String reason,
                            long committedAtMs,
                            long requestWaitMs) {
    @JsonProperty("committedAt")
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public String getCommittedAt() {
        return PvLogTimeFormatter.format(committedAtMs);
    }
}
