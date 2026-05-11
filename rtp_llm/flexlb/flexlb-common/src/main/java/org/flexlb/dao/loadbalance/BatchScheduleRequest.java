package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
public class BatchScheduleRequest {

    @JsonProperty("batch_count")
    private int batchCount;
}
