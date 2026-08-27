package org.flexlb.dao.route;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.flexlb.config.ServiceDiscoveryRuntimeConfig;
import org.flexlb.discovery.ServiceDiscoveryType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Service discovery configuration for one endpoint.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class DiscoveryConfig {

    public static final String DEFAULT_DASHSCOPE_BASE_URL = "http://127.0.0.1:8880";

    private static final ServiceDiscoveryRuntimeConfig DEFAULT_RUNTIME_CONFIG =
            new ServiceDiscoveryRuntimeConfig();

    @JsonProperty("type")
    private ServiceDiscoveryType type;

    @JsonProperty("base_url")
    private String baseUrl = DEFAULT_DASHSCOPE_BASE_URL;

    @JsonProperty("hosts")
    private List<String> hosts = new ArrayList<>();

    @JsonIgnore
    @EqualsAndHashCode.Exclude
    @ToString.Exclude
    private transient Supplier<ServiceDiscoveryRuntimeConfig> runtimeConfigSupplier =
            () -> DEFAULT_RUNTIME_CONFIG;

    /**
     * Bind runtime behavior after topology parsing. This keeps behavior out of
     * MODEL_SERVICE_CONFIG while preserving the provider-facing accessors.
     */
    @JsonIgnore
    public void bindRuntimeConfig(
            Supplier<ServiceDiscoveryRuntimeConfig> runtimeConfigSupplier) {
        this.runtimeConfigSupplier = Objects.requireNonNull(runtimeConfigSupplier);
    }

    @JsonIgnore
    public int getConnectTimeoutMs() {
        return runtimeConfig().getConnectTimeoutMs();
    }

    @JsonIgnore
    public int getReadTimeoutMs() {
        return runtimeConfig().getReadTimeoutMs();
    }

    @JsonIgnore
    public long getPollIntervalMs() {
        return runtimeConfig().getPollIntervalMs();
    }

    @JsonIgnore
    public int getMaxIdleConnections() {
        return runtimeConfig().getMaxIdleConnections();
    }

    @JsonIgnore
    public long getKeepAliveDurationMs() {
        return runtimeConfig().getKeepAliveDurationMs();
    }

    private ServiceDiscoveryRuntimeConfig runtimeConfig() {
        ServiceDiscoveryRuntimeConfig runtimeConfig = runtimeConfigSupplier.get();
        return runtimeConfig == null ? DEFAULT_RUNTIME_CONFIG : runtimeConfig;
    }
}
