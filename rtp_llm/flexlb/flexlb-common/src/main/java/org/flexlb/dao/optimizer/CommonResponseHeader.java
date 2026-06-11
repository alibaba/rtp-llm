package org.flexlb.dao.optimizer;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class CommonResponseHeader {

    private Status status;

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Status {
        private int code;
        private String message;
    }
}
