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

    @JsonProperty("dp_rank")
    private long dpRank;

    @JsonProperty("prefill_time")
    private long prefillTime;

    @JsonProperty("group")
    private String group;

    @JsonProperty("debug_info")
    private DebugInfo debugInfo;

    @JsonProperty("request_id")
    private String requestId;

    @JsonProperty("success")
    private boolean success;

    @JsonProperty("code")
    private int code;

    @JsonProperty("message")
    private String message;


    /** Return an independent copy, or null when the source is null. */
    public static ServerStatus copyOf(ServerStatus source) {
        if (source == null) {
            return null;
        }
        ServerStatus copy = new ServerStatus();
        copy.role = source.role;
        copy.serverIp = source.serverIp;
        copy.httpPort = source.httpPort;
        copy.grpcPort = source.grpcPort;
        copy.dpRank = source.dpRank;
        copy.prefillTime = source.prefillTime;
        copy.group = source.group;
        copy.debugInfo = DebugInfo.copyOf(source.debugInfo);
        copy.requestId = source.requestId;
        copy.success = source.success;
        copy.code = source.code;
        copy.message = source.message;
        return copy;
    }
}
