package org.flexlb.dao.route;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

/**
 * Online Optimizer trace-query configuration.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class OnlineOptimizerConfig {

    public static final String DEFAULT_PATH = "/api/optimizer";

    @JsonProperty("enabled")
    private boolean enabled;

    @JsonProperty("address")
    private String address;

    @JsonProperty("path")
    private String path = DEFAULT_PATH;

    @JsonProperty("discovery")
    private DiscoveryConfig discovery;

    public Endpoint toEndpoint() {
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress(address);
        endpoint.setProtocol("http");
        endpoint.setDiscovery(discovery);
        return endpoint;
    }

}
