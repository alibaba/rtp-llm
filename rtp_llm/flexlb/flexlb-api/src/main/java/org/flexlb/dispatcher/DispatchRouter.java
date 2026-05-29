package org.flexlb.dispatcher;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;

import java.util.List;

@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class DispatchRouter {

    private final GenericBatchHandler batchHandler;
    private final PassthroughClient passthroughClient;
    private final DispatcherInspectionHandler inspectionHandler;
    private final List<BatchEndpointSpec> specs;

    public DispatchRouter(GenericBatchHandler batchHandler,
                          PassthroughClient passthroughClient,
                          DispatcherInspectionHandler inspectionHandler,
                          List<BatchEndpointSpec> specs) {
        this.batchHandler = batchHandler;
        this.passthroughClient = passthroughClient;
        this.inspectionHandler = inspectionHandler;
        this.specs = specs;
    }

    public RouterFunction<ServerResponse> routes() {
        RouterFunctions.Builder b = RouterFunctions.route();
        for (BatchEndpointSpec spec : specs) {
            b.POST("/dispatcher" + spec.getPath(), req -> batchHandler.handle(req, spec));
        }
        b.GET("/dispatcher/_snapshot", inspectionHandler::snapshot);
        b.POST("/dispatcher/_dryrun/**", inspectionHandler::dryRun);
        return b.route(RequestPredicates.path("/dispatcher/**"), passthroughClient::forward)
                .build();
    }
}
