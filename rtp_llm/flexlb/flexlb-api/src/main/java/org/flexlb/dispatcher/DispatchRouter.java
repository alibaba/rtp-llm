package org.flexlb.dispatcher;

import org.flexlb.service.grace.ActiveRequestCounter;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.function.Supplier;

/**
 * Reactive {@code RouterFunction} for {@code /dispatcher/**}. Each registered batch endpoint
 * (see {@link org.flexlb.dispatcher.BatchEndpointSpec#SPECS}) gets a POST
 * route that delegates to {@link org.flexlb.dispatcher.BatchHandler};
 * {@code GET /dispatcher/_snapshot} and {@code POST /dispatcher/_dryrun/**} go to
 * {@link DispatcherInspectionHandler}; everything else under {@code /dispatcher/**} is
 * forwarded to one FE via {@link PassthroughClient#forward}.
 */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class DispatchRouter {

    private final BatchHandler batchHandler;
    private final PassthroughClient passthroughClient;
    private final DispatcherInspectionHandler inspectionHandler;
    private final ActiveRequestCounter activeRequestCounter;
    private final List<BatchEndpointSpec> specs;

    public DispatchRouter(BatchHandler batchHandler,
                          PassthroughClient passthroughClient,
                          DispatcherInspectionHandler inspectionHandler,
                          ActiveRequestCounter activeRequestCounter,
                          List<BatchEndpointSpec> specs) {
        this.batchHandler = batchHandler;
        this.passthroughClient = passthroughClient;
        this.inspectionHandler = inspectionHandler;
        this.activeRequestCounter = activeRequestCounter;
        this.specs = specs;
    }

    public RouterFunction<ServerResponse> routes() {
        RouterFunctions.Builder b = RouterFunctions.route();
        for (BatchEndpointSpec spec : specs) {
            b.POST("/dispatcher" + spec.getPath(), req -> tracked(() -> batchHandler.handle(req, spec)));
        }
        // Inspection endpoints are read-only diagnostics, not serving traffic — left out of the
        // graceful-drain count.
        b.GET("/dispatcher/_snapshot", inspectionHandler::snapshot);
        b.POST("/dispatcher/_dryrun/**", inspectionHandler::dryRun);
        return b.route(RequestPredicates.path("/dispatcher/**"), req -> tracked(() -> passthroughClient.forward(req)))
                .build();
    }

    /**
     * Counts the in-flight fanout / passthrough request against graceful-shutdown drain (mirrors
     * the {@code /schedule} path in {@code HttpLoadBalanceServer}) so a pre-stop drain waits for
     * it instead of dropping it mid-flight. The token is released on complete, error, and cancel.
     */
    private Mono<ServerResponse> tracked(Supplier<Mono<ServerResponse>> handler) {
        return Mono.using(activeRequestCounter::acquire, ignored -> handler.get(),
                ActiveRequestCounter.RequestToken::close);
    }
}
