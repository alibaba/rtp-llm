package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.dao.route.RoleType;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class ServerStatus {
    @JsonProperty("role")
    private RoleType role;

    @JsonProperty("server_ip")
    private String serverIp;

    @JsonProperty("http_port")
    private int httpPort;

    @JsonProperty("grpc_port")
    private int grpcPort;

    @JsonProperty("prefill_time")
    private long prefillTime;

    @JsonProperty("group")
    private String group;

    @JsonProperty("debug_info")
    private DebugInfo debugInfo;

    @JsonProperty("inter_request_id")
    private String interRequestId;

    @JsonProperty("success")
    private boolean success;

    @JsonProperty("code")
    private int code;

    @JsonProperty("message")
    private String message;

    public static ServerStatus code(StrategyErrorType code) {
        ServerStatus result = new ServerStatus();
        result.setSuccess(false);
        result.setCode(code.getErrorCode());
        result.setMessage(code.getErrorMsg());
        return result;
    }
}
