package org.flexlb.dispatcher;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;

@JsonIgnoreProperties(ignoreUnknown = true)
public class DispatchConfig {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @JsonProperty("enabled")
    private boolean enabled = false;

    @JsonProperty("dispatchPort")
    private int dispatchPort = 7005;

    @JsonProperty("subBatchSize")
    private int subBatchSize = 5;

    @JsonProperty("feRequestTimeoutMs")
    private int feRequestTimeoutMs = 3000;

    @JsonProperty("fePoolServiceId")
    private String fePoolServiceId = "";

    @JsonProperty("feMaxConnections")
    private int feMaxConnections = 200;

    @JsonProperty("feMaxPendingAcquire")
    private int feMaxPendingAcquire = 1000;

    @JsonProperty("feMaxResponseBytes")
    private int feMaxResponseBytes = 16 * 1024 * 1024;

    public static DispatchConfig fromJson(String json) {
        if (json == null || json.isBlank()) {
            return new DispatchConfig();
        }
        try {
            DispatchConfig c = MAPPER.readValue(json, DispatchConfig.class);
            c.validate();
            return c;
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalArgumentException("invalid DISPATCH_CONFIG: " + e.getMessage(), e);
        }
    }

    private void validate() {
        if (enabled && (fePoolServiceId == null || fePoolServiceId.isBlank())) {
            throw new IllegalArgumentException("DISPATCH_CONFIG.enabled=true requires fePoolServiceId");
        }
        if (subBatchSize < 1) {
            throw new IllegalArgumentException("subBatchSize must be >= 1");
        }
    }

    public boolean isEnabled() {
        return enabled;
    }

    public int getDispatchPort() {
        return dispatchPort;
    }

    public int getSubBatchSize() {
        return subBatchSize;
    }

    public int getFeRequestTimeoutMs() {
        return feRequestTimeoutMs;
    }

    public String getFePoolServiceId() {
        return fePoolServiceId;
    }

    public int getFeMaxConnections() {
        return feMaxConnections;
    }

    public int getFeMaxPendingAcquire() {
        return feMaxPendingAcquire;
    }

    public int getFeMaxResponseBytes() {
        return feMaxResponseBytes;
    }
}
