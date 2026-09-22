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

import java.net.URI;

/**
 * One allocated FE or BE address in server_status. A colocated PDFUSION worker's
 * HTTP port reaches its FE, and its gRPC port supplies BE preassignment.
 */
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

    public String httpUrl() {
        return "http://" + serverIp + ":" + httpPort;
    }

    public static BatchScheduleTarget frontend(String url) {
        URI uri = URI.create(url);
        BatchScheduleTarget target = new BatchScheduleTarget();
        target.serverIp = uri.getHost();
        target.httpPort = uri.getPort();
        target.role = RoleType.FRONTEND;
        return target;
    }

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
