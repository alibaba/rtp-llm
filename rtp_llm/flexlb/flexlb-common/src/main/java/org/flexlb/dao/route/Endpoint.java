package org.flexlb.dao.route;

import com.fasterxml.jackson.annotation.JsonIgnore;
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

    /**
     * Base TCP port of the per-engine worker-control gRPC endpoint. Engine with index i exposes
     * its worker status and cache status RPCs at {@code workerStatusPort + i}. Required when
     * {@code multiEngineNum > 1}; when unset for a single engine, discovery falls back to
     * the engine gRPC port. Validated at config load against the port range.
     */
    @JsonProperty("worker_status_port")
    private Integer workerStatusPort;

    /**
     * Number of engine processes sharing this physical endpoint (DS_LLM_MULTI_ENGINE_NUM).
     * Service discovery expands one endpoint into this many logical workers, each identified
     * by an engine index in {@code [0, multiEngineNum)} and its own worker status port.
     * Defaults to 1 for single-engine deployments.
     */
    @JsonProperty("multi_engine_num")
    private int multiEngineNum = 1;

    @JsonProperty("discovery")
    private DiscoveryConfig discovery;

    @JsonIgnore
    private String group = "";
}
