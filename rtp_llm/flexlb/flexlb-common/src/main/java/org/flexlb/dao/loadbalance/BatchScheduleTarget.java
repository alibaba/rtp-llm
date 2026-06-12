package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;
import org.flexlb.dao.route.RoleType;

/**
 * One scheduled worker target. Exactly one of {@link #grpcPort}/{@link #arpcPort} is set,
 * decided by the configured {@link org.flexlb.enums.EngineType}: LLM engines serve gRPC on
 * registered port + 1, embedding engines serve ARPC there. The unset slot is omitted from
 * JSON, so a caller never sees a port labeled with a protocol the worker does not speak.
 */
@Getter
@Setter
@ToString
@NoArgsConstructor
@AllArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
public class BatchScheduleTarget {

    @JsonProperty("server_ip")
    private String serverIp;

    @JsonProperty("http_port")
    private int httpPort;

    /** Engine gRPC port (LLM engines only). */
    @JsonProperty("grpc_port")
    private Integer grpcPort;

    /** Engine ARPC port, MainseBertRpcService (embedding/BERT engines only). */
    @JsonProperty("arpc_port")
    private Integer arpcPort;

    /**
     * Role this target serves. Mirrors Python {@code RoleAddr.role} so the dispatcher can stamp
     * {@code generate_config.role_addrs} 1:1 from {@link BatchScheduleTarget} without remapping.
     * Carried per-target rather than at response level so a future multi-role batch_schedule
     * (mixed PREFILL+DECODE in one response) needs no schema change.
     */
    @JsonProperty("role")
    private RoleType role;

    /** Convenience for callers that build targets without a role (older tests, error paths). */
    public BatchScheduleTarget(String serverIp, int httpPort, Integer grpcPort) {
        this.serverIp = serverIp;
        this.httpPort = httpPort;
        this.grpcPort = grpcPort;
    }

    /** Convenience for LLM-flavored callers and tests. */
    public BatchScheduleTarget(String serverIp, int httpPort, Integer grpcPort, RoleType role) {
        this.serverIp = serverIp;
        this.httpPort = httpPort;
        this.grpcPort = grpcPort;
        this.role = role;
    }
}
