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

    /** Whether the response must contain backend worker fields. */
    private boolean assignBe = true;

    /** Whether the elected master should stamp {@code fe_url}. */
    private boolean assignFe = true;
}
