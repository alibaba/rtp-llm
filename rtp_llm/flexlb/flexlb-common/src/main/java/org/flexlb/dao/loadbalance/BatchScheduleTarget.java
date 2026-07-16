package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.CommonUtils;

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

    /**
     * Test-only convenience for a role-less gRPC target. Production builds targets via
     * {@link #of(WorkerStatus, RoleType, boolean)} or the no-arg constructor plus setters.
     */
    public BatchScheduleTarget(String serverIp, int httpPort, Integer grpcPort) {
        this.serverIp = serverIp;
        this.httpPort = httpPort;
        this.grpcPort = grpcPort;
    }

    /**
     * Test-only convenience for a role-tagged gRPC target. See
     * {@link #of(WorkerStatus, RoleType, boolean)} for the production path.
     */
    public BatchScheduleTarget(String serverIp, int httpPort, Integer grpcPort, RoleType role) {
        this.serverIp = serverIp;
        this.httpPort = httpPort;
        this.grpcPort = grpcPort;
        this.role = role;
    }

    /**
     * Build a target for {@code worker} in {@code roleType}, filling exactly the one port slot the
     * engine speaks: ARPC for embedding engines, gRPC for LLM. This is the single home for the
     * "registered port + 1, protocol by engine type" rule (mirroring {@link ServerStatus#ok}), so a
     * selection strategy never has to re-derive it.
     */
    public static BatchScheduleTarget of(WorkerStatus worker, RoleType roleType, boolean embedding) {
        BatchScheduleTarget target = new BatchScheduleTarget();
        target.serverIp = worker.getIp();
        target.httpPort = worker.getPort();
        if (embedding) {
            target.arpcPort = CommonUtils.toArpcPort(worker.getPort());
        } else {
            target.grpcPort = CommonUtils.toGrpcPort(worker.getPort());
        }
        target.role = roleType;
        return target;
    }
}
