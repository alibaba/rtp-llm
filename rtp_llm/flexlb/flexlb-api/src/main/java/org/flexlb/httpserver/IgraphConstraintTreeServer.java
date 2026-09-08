package org.flexlb.httpserver;

import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.constraint.IgraphConstraintTreePoller;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import java.util.Map;

import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Component
@ConditionalOnProperty(name = "constraint.tree.igraph.enabled", havingValue = "true")
public class IgraphConstraintTreeServer {
    private final IgraphConstraintTreePoller poller;
    private final LBStatusConsistencyService consistency;

    public IgraphConstraintTreeServer(IgraphConstraintTreePoller poller, LBStatusConsistencyService consistency) {
        this.poller = poller;
        this.consistency = consistency;
    }

    @Bean
    public RouterFunction<ServerResponse> igraphConstraintTreeRoutes() {
        return route()
                .GET("/rtp_llm/constraint_tree/source/status", request ->
                        ServerResponse.ok().bodyValue(poller.getStatus()))
                .POST("/rtp_llm/constraint_tree/source/refresh", request -> {
                    if (!consistency.isMaster()) {
                        return ServerResponse.status(503).bodyValue(Map.of("error", "send refresh to active Master"));
                    }
                    boolean accepted = poller.trigger();
                    return ServerResponse.status(accepted ? 202 : 409).bodyValue(Map.of("accepted", accepted,
                            "message", accepted ? "refresh queued; not yet published" : "busy, not ready or shutting down"));
                }).build();
    }
}
