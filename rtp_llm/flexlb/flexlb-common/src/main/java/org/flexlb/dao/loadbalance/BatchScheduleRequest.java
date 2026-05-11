package org.flexlb.dao.loadbalance;

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
public class BatchScheduleRequest {

    @JsonProperty("batch_count")
    private int batchCount;

    @JsonProperty("sub_requests")
    private List<Request> subRequests;
}
