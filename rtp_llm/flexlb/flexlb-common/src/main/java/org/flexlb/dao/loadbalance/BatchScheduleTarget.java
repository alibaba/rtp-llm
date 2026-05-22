package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;
import org.flexlb.dao.route.RoleType;

@Getter
@Setter
@ToString
@NoArgsConstructor
@AllArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class BatchScheduleTarget {

    @JsonProperty("server_ip")
    private String serverIp;

    @JsonProperty("http_port")
    private int httpPort;

    @JsonProperty("grpc_port")
    private int grpcPort;

    /**
     * Role this target serves. Mirrors Python {@code RoleAddr.role} so the dispatcher can stamp
     * {@code generate_config.role_addrs} 1:1 from {@link BatchScheduleTarget} without remapping.
     * Carried per-target rather than at response level so a future multi-role batch_schedule
     * (mixed PREFILL+DECODE in one response) needs no schema change.
     */
    @JsonProperty("role")
    private RoleType role;

    /** Convenience for callers that build targets without a role (older tests, error paths). */
    public BatchScheduleTarget(String serverIp, int httpPort, int grpcPort) {
        this.serverIp = serverIp;
        this.httpPort = httpPort;
        this.grpcPort = grpcPort;
    }
}
