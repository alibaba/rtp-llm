package org.flexlb.dao.route;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.discovery.ServiceDiscoveryType;

import java.util.ArrayList;
import java.util.List;

/**
 * Service discovery configuration for one endpoint.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class DiscoveryConfig {

    public static final String DEFAULT_DASHSCOPE_BASE_URL = "http://127.0.0.1:8880";

    @JsonProperty("type")
    private ServiceDiscoveryType type;

    @JsonProperty("base_url")
    private String baseUrl = DEFAULT_DASHSCOPE_BASE_URL;

    @JsonProperty("connect_timeout_ms")
    private int connectTimeoutMs = 500;

    @JsonProperty("read_timeout_ms")
    private int readTimeoutMs = 500;

    @JsonProperty("poll_interval_ms")
    private long pollIntervalMs = 1000L;

    @JsonProperty("max_idle_connections")
    private int maxIdleConnections = 5;

    @JsonProperty("keep_alive_duration_ms")
    private long keepAliveDurationMs = 300000L;

    @JsonProperty("hosts")
    private List<String> hosts = new ArrayList<>();
}
