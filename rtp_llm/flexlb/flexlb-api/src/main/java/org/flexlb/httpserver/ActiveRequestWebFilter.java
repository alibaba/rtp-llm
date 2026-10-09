package org.flexlb.httpserver;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import org.springframework.web.server.WebFilter;
import org.springframework.web.server.WebFilterChain;
import reactor.core.publisher.Mono;

/** Drains serving HTTP exchanges without waiting on the pre-stop hook that initiates shutdown. */
@Component
public final class ActiveRequestWebFilter implements WebFilter {
    private int activeRequests;
    private boolean draining;

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, WebFilterChain chain) {
        String path = exchange.getRequest().getPath().pathWithinApplication().value();
        boolean serving = path.equals("/rtp_llm/batch_schedule")
                || path.equals("/dispatcher")
                || (path.startsWith("/dispatcher/") && !path.equals("/dispatcher/_dryrun") && !path.startsWith("/dispatcher/_dryrun/"));
        if (!serving) {
            return chain.filter(exchange);
        }
        return Mono.defer(() -> {
            synchronized (this) {
                if (draining) {
                    exchange.getResponse().setStatusCode(HttpStatus.SERVICE_UNAVAILABLE);
                    return exchange.getResponse().setComplete();
                }
                activeRequests++;
            }
            return Mono.defer(() -> chain.filter(exchange))
                    .doFinally(ignored -> requestFinished());
        });
    }

    private synchronized void requestFinished() {
        if (--activeRequests == 0) {
            notifyAll();
        }
    }

    /** Stop new work before waiting; serving dependencies stay alive until accepted HTTP work finishes. */
    public synchronized void drain() {
        draining = true;
        boolean interrupted = false;
        try {
            while (activeRequests != 0) {
                try {
                    wait();
                } catch (InterruptedException e) {
                    // Match gRPC drain: only the platform's forced-kill deadline ends active work.
                    interrupted = true;
                }
            }
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
