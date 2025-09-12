package org.flexlb.domain.batch;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class SubmitBatchResponse {

    @JsonProperty("code")
    private Integer code;

    @JsonProperty("message")
    private String message;

    public static SubmitBatchResponse error(String message) {
        SubmitBatchResponse response = new SubmitBatchResponse();
        response.setCode(500);
        response.setMessage(message);
        return response;
    }

    public static SubmitBatchResponse success() {
        SubmitBatchResponse response = new SubmitBatchResponse();
        response.setCode(200);
        response.setMessage("success");
        return response;
    }
}
