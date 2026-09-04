package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
@JsonIgnoreProperties(ignoreUnknown = true)
public class BatchScheduleRequest {

    @JsonProperty("batch_count")
    private int batchCount;

    /**
     * Whether the response must contain backend worker fields. Defaults to {@code true} so an old
     * caller that only sends {@code batch_count} keeps the original wire behavior. A dispatcher
     * endpoint that cannot consume {@code role_addrs} sends {@code false}; the master then returns
     * placeholder targets for FE stamping without advancing a backend strategy cursor.
     */
    @JsonProperty("assign_be")
    private boolean assignBe = true;

    /**
     * Whether the elected master should stamp {@code fe_url}. Defaults to {@code true} for wire
     * compatibility. Dispatcher local-FE mode sends {@code false}, avoiding an unused advance of
     * the master's FE cursor when the same request still needs BE pre-assignment.
     */
    @JsonProperty("assign_fe")
    private boolean assignFe = true;
}
