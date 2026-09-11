package org.flexlb.httpserver;

import lombok.RequiredArgsConstructor;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import org.springframework.web.server.WebFilter;
import org.springframework.web.server.WebFilterChain;
import reactor.core.publisher.Mono;

/** Counts serving requests until their HTTP exchange completes, fails, or is cancelled. */
@Component
@RequiredArgsConstructor
public final class ActiveRequestWebFilter implements WebFilter {
    private final ActiveRequestCounter activeRequests;

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, WebFilterChain chain) {
        String path = exchange.getRequest().getPath().pathWithinApplication().value();
        boolean serving = path.equals("/rtp_llm/batch_schedule")
                || path.equals("/dispatcher") || path.startsWith("/dispatcher/");
        if (!serving) {
            return chain.filter(exchange);
        }
        return Mono.using(activeRequests::acquire,
                token -> Mono.defer(() -> chain.filter(exchange)),
                ActiveRequestCounter.RequestToken::close);
    }
}
