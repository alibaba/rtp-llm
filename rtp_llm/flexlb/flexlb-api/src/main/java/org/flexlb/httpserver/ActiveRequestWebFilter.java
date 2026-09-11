package org.flexlb.httpserver;

import org.flexlb.service.grace.ActiveRequestCounter;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import org.springframework.web.server.WebFilter;
import org.springframework.web.server.WebFilterChain;
import reactor.core.publisher.Mono;

/** Counts serving requests until their HTTP exchange completes, fails, or is cancelled. */
@Component
public final class ActiveRequestWebFilter implements WebFilter {
    private final ActiveRequestCounter activeRequests;

    public ActiveRequestWebFilter(ActiveRequestCounter activeRequests) {
        this.activeRequests = activeRequests;
    }

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, WebFilterChain chain) {
        String path = exchange.getRequest().getPath().pathWithinApplication().value();
        boolean serving = path.equals("/rtp_llm/batch_schedule")
                || ((path.equals("/dispatcher") || path.startsWith("/dispatcher/"))
                    && !path.equals("/dispatcher/_snapshot")
                    && !path.startsWith("/dispatcher/_dryrun/"));
        if (!serving) {
            return chain.filter(exchange);
        }
        return Mono.using(activeRequests::acquire,
                token -> Mono.defer(() -> chain.filter(exchange)),
                ActiveRequestCounter.RequestToken::close);
    }
}
