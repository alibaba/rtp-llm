package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
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
    @JsonInclude(JsonInclude.Include.NON_DEFAULT)
    private int httpPort;

    /** Engine gRPC port (LLM engines only). */
    @JsonProperty("grpc_port")
    private Integer grpcPort;

    /** Engine ARPC port, MainseBertRpcService (embedding/BERT engines only). */
    @JsonProperty("arpc_port")
    private Integer arpcPort;

    /** Backend role consumed by the FE's role_addrs field. */
    @JsonProperty("role")
    private RoleType role;

    /**
     * Optional master-assigned FE base URL ({@code http://ip:port}) for this chunk. Present when
     * the request carries {@code assign_fe=true}; {@code null} for BE-only and FE-only placeholder
     * responses the master could not stamp. A dispatcher in master mode treats a missing value as
     * a visible chunk failure. A dispatcher in local mode ignores this field and reserves its own
     * index-aligned FE vector.
     */
    @JsonProperty("fe_url")
    private String feUrl;

    public BatchScheduleTarget(String serverIp, int httpPort, Integer grpcPort) {
        this.serverIp = serverIp;
        this.httpPort = httpPort;
        this.grpcPort = grpcPort;
    }

    public BatchScheduleTarget(String serverIp, int httpPort, Integer grpcPort, RoleType role) {
        this.serverIp = serverIp;
        this.httpPort = httpPort;
        this.grpcPort = grpcPort;
        this.role = role;
    }

    /** The registered HTTP port determines the engine's RPC port and protocol. */
    public static BatchScheduleTarget of(WorkerHost worker, RoleType role, EngineType engineType) {
        BatchScheduleTarget target = new BatchScheduleTarget();
        target.serverIp = worker.getIp();
        target.httpPort = worker.getHttpPort();
        if (engineType == EngineType.EMBEDDING) {
            target.arpcPort = CommonUtils.toArpcPort(worker.getHttpPort());
        } else {
            target.grpcPort = CommonUtils.toGrpcPort(worker.getHttpPort());
        }
        target.role = role;
        return target;
    }
}
