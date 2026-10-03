package org.flexlb.dao.optimizer;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class CommonResponseHeader {

    private Status status;

    @JsonProperty("request_id")
    private String requestId;

    @JsonProperty("tracer_result")
    private String tracerResult;

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Status {
        private OptimizerErrorCode code;
        private String message;
    }
}
