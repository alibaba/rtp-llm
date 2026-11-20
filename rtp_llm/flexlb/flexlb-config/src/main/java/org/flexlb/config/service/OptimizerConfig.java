package org.flexlb.config.service;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.util.discovery.DiscoveryEndpoint;
import org.flexlb.util.discovery.DiscoverySettings;

/**
 * Optimizer trace-query configuration.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class OptimizerConfig {

    public static final String DEFAULT_PATH = "/api/optimizer";
    public static final int DEFAULT_PORT = 8082;

    @JsonProperty("address")
    private String address;

    /**
     * Optimizer HTTP port overriding the port returned by service discovery.
     */
    @JsonProperty("port")
    private int port = DEFAULT_PORT;

    @JsonProperty("path")
    private String path = DEFAULT_PATH;

    @JsonProperty("discovery")
    private DiscoverySettings discovery;

    public DiscoveryEndpoint toEndpoint() {
        DiscoveryEndpoint endpoint = new DiscoveryEndpoint();
        endpoint.setAddress(address);
        endpoint.setProtocol("http");
        endpoint.setDiscovery(discovery);
        return endpoint;
    }
}
