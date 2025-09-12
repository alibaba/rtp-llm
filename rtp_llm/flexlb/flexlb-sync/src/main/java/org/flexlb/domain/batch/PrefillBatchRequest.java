package org.flexlb.domain.batch;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class PrefillBatchRequest {

    @JsonProperty("ip")
    private String ip;

    @JsonProperty("port")
    private int port;

    @JsonProperty("model")
    private String model;

    @JsonProperty("batch_id")
    private String batchId;

    @JsonProperty("requests")
    private List<Object> requests;
}
