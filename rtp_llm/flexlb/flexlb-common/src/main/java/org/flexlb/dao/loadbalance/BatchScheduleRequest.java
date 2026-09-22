package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
public class BatchScheduleRequest {

    private int batchCount;

    /** Number of master forwards; omitted by existing callers and therefore initially zero. */
    private int forwardHop;

    /** Existing RTP callers request workers only; Dispatcher explicitly requests FEs. */
    private AllocationType allocationType = AllocationType.BE;

    public enum AllocationType {
        BE, FE, FE_AND_BE;

        public boolean includesBe() {
            return this == BE || this == FE_AND_BE;
        }

        public boolean includesFe() {
            return this == FE || this == FE_AND_BE;
        }
    }
}
