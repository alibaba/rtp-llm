package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AccessLevel;
import lombok.Data;
import lombok.Setter;
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

    /**
     * Selected logical engine index exposed in the schedule response. This field is present
     * when a physical frontend owns multiple logical engines and omitted when there is only
     * one engine.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    @JsonProperty("engine_index")
    private Integer engineIndex;

    /**
     * Selected logical engine index used only inside FlexLB. Unlike {@link #engineIndex}, this
     * value is always available, including {@code 0} for a single-engine frontend, so routing
     * and rollback can address the exact logical worker.
     */
    @JsonIgnore
    @Setter(AccessLevel.NONE)
    private int routingEngineIndex;

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

    public static ServerStatus code(StrategyErrorType code) {
        ServerStatus result = new ServerStatus();
        result.setSuccess(false);
        result.setCode(code.getErrorCode());
        result.setMessage(code.getErrorMsg());
        return result;
    }

    /**
     * Sets the selected engine identity for internal routing and the schedule wire response.
     * Single-engine routes keep the internal {@code @0} identity but omit the wire field.
     */
    public void setSelectedEngineIndex(int selectedEngineIndex, int multiEngineNum) {
        if (multiEngineNum < 1
                || selectedEngineIndex < 0
                || selectedEngineIndex >= multiEngineNum) {
            throw new IllegalArgumentException(
                    "selected engine index must be in [0, multiEngineNum)");
        }
        routingEngineIndex = selectedEngineIndex;
        engineIndex = multiEngineNum > 1 ? selectedEngineIndex : null;
    }

    /**
     * Returns the internal logical worker identity in {@code ip:port@engineIndex} format.
     * The index identifies one independently routable engine behind the physical frontend.
     */
    @JsonIgnore
    public String getLogicalIpPort() {
        return serverIp + ":" + httpPort + "@" + routingEngineIndex;
    }
}
