package org.flexlb.dispatcher;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.util.JsonUtils;

@Getter
@Setter
public class DispatchConfig {

    private boolean enabled = false;

    private int subBatchSize = 5;

    private int feRequestTimeoutMs = 3000;

    private int feConnectTimeoutMs = 2000;

    private int feResponseTimeoutMs = 5000;

    private int feMaxStreamDurationMs = 600_000;

    private String fePoolServiceId = "";

    private int feMaxConnections = 200;

    private int feMaxPendingAcquire = 1000;

    private int feMaxResponseBytes = 16 * 1024 * 1024;

    public static DispatchConfig fromJson(String json) {
        if (json == null || json.isBlank()) {
            return new DispatchConfig();
        }
        DispatchConfig c = JsonUtils.toObject(json, DispatchConfig.class);
        c.validate();
        return c;
    }

    private void validate() {
        if (enabled && (fePoolServiceId == null || fePoolServiceId.isBlank())) {
            throw new IllegalArgumentException("DISPATCH_CONFIG.enabled=true requires fePoolServiceId");
        }
        if (subBatchSize < 1) {
            throw new IllegalArgumentException("subBatchSize must be >= 1, got " + subBatchSize);
        }
    }
}
