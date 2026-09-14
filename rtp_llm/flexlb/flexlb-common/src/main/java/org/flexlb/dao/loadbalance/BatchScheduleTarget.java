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

/** Scheduled worker target: EMBEDDING exposes ARPC, LLM exposes gRPC. */
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

    private Integer grpcPort;

    private Integer arpcPort;

    private RoleType role;

    private String feUrl;

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
