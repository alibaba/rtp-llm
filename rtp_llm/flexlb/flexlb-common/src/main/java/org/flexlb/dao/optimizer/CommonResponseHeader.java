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
        // Object type to accept both protobuf enum string ("OK") and integer (1)
        private Object code;
        private String message;

        public boolean isOk() {
            if (code == null) return false;
            if (code instanceof Number) return ((Number) code).intValue() == 1;
            return "OK".equals(code);
        }
    }
}
