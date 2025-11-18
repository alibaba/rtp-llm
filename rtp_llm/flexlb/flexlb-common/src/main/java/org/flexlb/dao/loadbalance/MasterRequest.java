package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

/**
 * @author zjw
 * description:
 * date: 2025/3/11
 */
@Getter
@Setter
public class MasterRequest {

    @JsonProperty("model")
    private String model;

    @JsonProperty("block_cache_keys")
    private List<Long> blockCacheKeys;

    @JsonProperty("seq_len")
    private long seqLen;
}
