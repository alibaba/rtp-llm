package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
import org.flexlb.util.CommonUtils;

/** One scheduled worker target. */
@Getter
@Setter
@ToString
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
@JsonInclude(JsonInclude.Include.NON_NULL)
public class BatchScheduleTarget {

    private String serverIp;

    @JsonInclude(JsonInclude.Include.NON_DEFAULT)
    private int httpPort;

    /** Engine gRPC port (LLM engines only). */
    private Integer grpcPort;

    /** Engine ARPC port, MainseBertRpcService (embedding/BERT engines only). */
    private Integer arpcPort;

    /** Backend role consumed by the FE's role_addrs field. */
    private RoleType role;

    /** FE URL assigned by the master when assign_fe is requested; absent for BE-only placement. */
    private String feUrl;

    /** The registered HTTP port determines the engine's RPC port and protocol. */
    public static BatchScheduleTarget of(WorkerHost worker, RoleType role, EngineType engineType) {
        BatchScheduleTarget target = new BatchScheduleTarget();
        target.serverIp = worker.getIp();
        target.httpPort = worker.getHttpPort();
        if (engineType == EngineType.EMBEDDING) {
            target.arpcPort = worker.getHttpPort() + 1;
        } else {
            target.grpcPort = CommonUtils.toGrpcPort(worker.getHttpPort());
        }
        target.role = role;
        return target;
    }
}
