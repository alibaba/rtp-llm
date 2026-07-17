package org.flexlb.dispatcher;

import org.flexlb.service.grace.ActiveRequestCounter;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseCookie;
import org.springframework.stereotype.Component;
import org.springframework.util.MultiValueMap;
import org.springframework.web.reactive.function.server.RequestPredicates;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;
import org.springframework.web.server.ServerWebExchange;
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
        return b.route(RequestPredicates.path("/dispatcher/**"), req -> tracked(() -> passthroughClient.forward(req)))
                .build();
    }

    /**
     * Counts the in-flight fanout / passthrough request against graceful-shutdown drain (mirrors
     * the {@code /schedule} path in {@code HttpLoadBalanceServer}) so a pre-stop drain waits for
     * it instead of dropping it mid-flight. The token is held until the response body has been
     * fully written, not merely until the {@code ServerResponse} is produced: a streaming
     * passthrough emits its {@code ServerResponse} the instant FE headers arrive but keeps writing
     * the body for the lifetime of the stream, so releasing at {@code ServerResponse} time would
     * let the drain declare the request finished and shut down mid-stream. The token is released
     * exactly once — on body completion, error, or cancel (client disconnect), or if the handler
     * never produces a response.
     */
    private Mono<ServerResponse> tracked(Supplier<Mono<ServerResponse>> handler) {
        return Mono.fromSupplier(activeRequestCounter::acquire).flatMap(token ->
                        Mono.defer(handler::get)
                                .<ServerResponse>map(response -> new DrainTrackedResponse(response, token))
                                .switchIfEmpty(Mono.<ServerResponse>fromRunnable(token::close))
                                .doOnError(e -> token.close())
                                .doOnCancel(token::close))
                // Belt and braces for the emitted-but-never-written window: a client disconnect
                // after the handler produced the response but before WebFlux subscribes writeTo
                // drops the wrapped response, and then writeTo's doFinally never fires. A cancel
                // that reaches this chain still fires doOnCancel above, but a drop surfaced only
                // as a discard (e.g. an operator clearing a held value) must close the token too.
                // Token close is idempotent, so overlapping paths cannot double-release.
                .doOnDiscard(DrainTrackedResponse.class, DrainTrackedResponse::closeToken);
    }

    /**
     * Delegating {@link ServerResponse} that defers releasing the graceful-drain token until the
     * response body has finished writing ({@code writeTo} completes, errors, or is cancelled), so
     * a still-streaming response stays counted against the drain.
     */
    private static final class DrainTrackedResponse implements ServerResponse {

        private final ServerResponse delegate;
        private final ActiveRequestCounter.RequestToken token;

        DrainTrackedResponse(ServerResponse delegate, ActiveRequestCounter.RequestToken token) {
            this.delegate = delegate;
            this.token = token;
        }

        /** Releases the drain token when this response is discarded before {@code writeTo} ran. */
        void closeToken() {
            token.close();
        }

        @Override
        public HttpStatus statusCode() {
            return delegate.statusCode();
        }

        @Override
        public int rawStatusCode() {
            return delegate.rawStatusCode();
        }

        @Override
        public HttpHeaders headers() {
            return delegate.headers();
        }

        @Override
        public MultiValueMap<String, ResponseCookie> cookies() {
            return delegate.cookies();
        }

        @Override
        public Mono<Void> writeTo(ServerWebExchange exchange, Context context) {
            return delegate.writeTo(exchange, context).doFinally(signal -> token.close());
        }
    }
}
