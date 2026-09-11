package org.flexlb.dispatcher;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;

import java.util.List;

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
    private final List<BatchEndpointSpec> specs;

    public DispatchRouter(BatchHandler batchHandler,
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
            String path = "/dispatcher" + spec.getPath();
            b.POST(path, req -> batchHandler.handle(req, spec));
            // The root spec's path is "/", so the loop above only registers "/dispatcher/". A
            // caller posting to "/dispatcher" would miss every batch route and be silently
            // passthrough-forwarded to a single FE unsplit — register the bare form too.
            if (path.endsWith("/")) {
                String bare = path.substring(0, path.length() - 1);
                b.POST(bare, req -> batchHandler.handle(req, spec));
            }
        }
        // Diagnostics, not serving traffic — left out of the graceful-drain count.
        //
        // They sit unauthenticated on the 7001 listener because *nothing* here is authenticated:
        // there is no security dependency and no auth filter in the module, and the sibling
        // endpoints on this same listener already expose the equivalent surface — /rtp_llm/
        // schedule_snapshot dumps the whole LB view (worker addresses + load), and
        // /rtp_llm/update_log_level and /rtp_llm/notify_master additionally mutate state.
        // 7001 is a trusted internal port by platform design; gating only these two would leave
        // the same information a request away, so it buys nothing.
        //
        // Note the management listener (7002) is not a safer home: it runs Actuator with
        // `base-path: /` and `exposure.include: "*"`, also unauthenticated, so /env and
        // /configprops there expose strictly more than this does. If the trusted-port assumption
        // ever stops holding, it stops holding for the whole listener and has to be fixed there,
        // not per-endpoint.
        //
        // What is genuinely this handler's business — a diagnostic must not perturb production —
        // is enforced in DispatcherInspectionHandler: dry-run resolves no BE targets (and so
        // never advances master's round-robin cursor) unless explicitly asked with
        // ?pre_assign=true.
        b.GET("/dispatcher/_snapshot", inspectionHandler::snapshot);
        b.POST("/dispatcher/_dryrun/**", inspectionHandler::dryRun);
        return b.route(RequestPredicates.path("/dispatcher/**"), req -> passthroughClient.forward(req)
                        .doOnDiscard(ServerResponse.class, PassthroughClient::releaseIfUnwritten))
                .build();
    }

}
