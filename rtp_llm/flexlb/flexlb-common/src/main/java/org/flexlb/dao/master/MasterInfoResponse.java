package org.flexlb.dao.master;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.dao.loadbalance.Response;

import java.util.Map;

/**
 * Master topology returned to FlexLB clients.
 */
@Data
public class MasterInfoResponse {

    @JsonProperty("success")
    private boolean success = true;

    @JsonProperty("code")
    private int code = 200;

    @JsonProperty("real_master_host")
    private String realMasterHost;

    @JsonProperty("pod_ip")
    private String podIp;

    @JsonProperty("instance_ip")
    private String instanceIp;

    @JsonProperty("queue_length")
    private int queueLength;

    @JsonProperty("worker_summary")
    private Map<String, Response.WorkerRoleSummary> workerSummary;

    @JsonProperty("ready")
    private boolean ready = true;
}
