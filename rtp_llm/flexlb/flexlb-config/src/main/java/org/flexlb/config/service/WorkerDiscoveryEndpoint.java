package org.flexlb.config.service;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;
import org.flexlb.util.discovery.DiscoveryEndpoint;

/**
 * Adds Worker deployment ports, Engine count and routing group to provider discovery settings.
 */
@Getter
@Setter
@lombok.EqualsAndHashCode(callSuper = true)
@lombok.ToString(callSuper = true)
public class WorkerDiscoveryEndpoint extends DiscoveryEndpoint {
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
     * Worker address resolution expands one endpoint into this many logical workers, each identified
     * by an engine index in {@code [0, multiEngineNum)} and its own worker status port.
     * Defaults to 1 for single-engine deployments.
     */
    @JsonProperty("multi_engine_num")
    private int multiEngineNum = 1;

    @JsonIgnore private String group = "";
}
