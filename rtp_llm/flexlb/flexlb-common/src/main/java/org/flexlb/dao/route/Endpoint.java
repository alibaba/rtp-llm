package org.flexlb.dao.route;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class Endpoint {

    @JsonProperty("address")
    private String address;

    @JsonProperty("protocol")
    private String protocol;

    @JsonProperty("path")
    private String path;
}
