package org.flexlb.util.discovery;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

/**
 * Describes a provider endpoint and the protocol used by its consumer.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class DiscoveryEndpoint {

    @JsonProperty("address")
    private String address;

    @JsonProperty("protocol")
    private String protocol;

    @JsonProperty("path")
    private String path;

    @JsonProperty("discovery")
    private DiscoverySettings discovery;
}
