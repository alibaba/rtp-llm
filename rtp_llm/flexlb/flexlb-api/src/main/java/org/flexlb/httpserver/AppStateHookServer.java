package org.flexlb.httpserver;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.service.grace.GracefulLifecycleReporter;
import org.flexlb.service.grace.GracefulOnlineService;
import org.flexlb.service.grace.GracefulShutdownService;
import org.flexlb.service.grace.strategy.ActiveRequestShutdownHooker;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.annotation.Bean;
import org.springframework.context.event.EventListener;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.net.InetAddress;

import static org.springframework.web.reactive.function.server.RequestPredicates.accept;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Slf4j
@Data
@Component
public class AppStateHookServer {

    private final GracefulOnlineService gracefulOnlineService;
    private final GracefulShutdownService gracefulShutdownService;
    private final GracefulLifecycleReporter lifecycleReporter;

    private volatile boolean processReady = false;

    private volatile InetAddress localAddress;

    public AppStateHookServer(GracefulOnlineService gracefulOnlineService,
                              GracefulShutdownService gracefulShutdownService,
                              GracefulLifecycleReporter lifecycleReporter) {
        this.gracefulOnlineService = gracefulOnlineService;
        this.gracefulShutdownService = gracefulShutdownService;
        this.lifecycleReporter = lifecycleReporter;
    }

    @EventListener(ApplicationReadyEvent.class)
    public void onApplicationReady() {
        try {
            localAddress = InetAddress.getLocalHost();
        } catch (Exception e) {
            log.warn("Failed to get local host address", e);
        }
        processReady = true;
    }

    @Bean
    public RouterFunction<ServerResponse> appStateHook() {
        return route()
                .GET("/hook/process_ok", accept(MediaType.ALL), this::handleProcessOk)
                .GET("/hook/after_start", accept(MediaType.ALL), this::handleAppStartUp)
                .GET("/hook/pre_stop", accept(MediaType.ALL), this::handleAppStop)
                .build();
    }

    public Mono<ServerResponse> handleProcessOk(ServerRequest request) {
        log.warn("recv /hook/process_ok request.");
        if (isNonLocalRequest(request)) {
            return buildErrorResponse("handleProcessOk");
        }
        if (processReady) {
            log.info("handleProcessOk success.");
            lifecycleReporter.reportProcessOk();
            return ServerResponse.ok().body(Mono.just("ok"), String.class);
        } else {
            return ServerResponse.status(503).body(Mono.just("not ready"), String.class);
        }
    }

    // NOTE: Intentionally blocking the Reactor event loop here.
    // This hook is invoked only once during startup by the local sidecar,
    // before any traffic is served. Blocking ensures the 200 response is
    // returned only after online() completes, so the sidecar can reliably
    // determine startup success. No need to offload to boundedElastic.
    public Mono<ServerResponse> handleAppStartUp(ServerRequest request) {
        log.warn("recv /hook/after_start request.");
        if (isNonLocalRequest(request)) {
            return buildErrorResponse("handleAppStartUp");
        }
        long startTime = System.currentTimeMillis();
        try {
            gracefulOnlineService.online();
            long duration = System.currentTimeMillis() - startTime;
            lifecycleReporter.reportOnlineComplete(duration);
            log.info("online service run success.");
            return ServerResponse.ok().body(Mono.just("success"), String.class);
        } catch (Exception e) {
            log.warn("handleOnline error.", e);
            return ServerResponse.status(500).body(Mono.just("error"), String.class);
        }
    }

    public Mono<ServerResponse> handleAppStop(ServerRequest request) {
        log.warn("recv /hook/pre_stop request.");

        if (isNonLocalRequest(request)) {
            return buildErrorResponse("handleAppStop");
        }
        return Mono.fromRunnable(gracefulShutdownService::offline)
                .subscribeOn(Schedulers.boundedElastic())
                .then(Mono.defer(() -> {
                    if (ActiveRequestShutdownHooker.shutdownCompletedSuccessfully) {
                        log.info("shutDown Service offline success, active request complete.");
                        return ServerResponse.ok().body(Mono.just("Shutdown complete. Ready for termination."), String.class);
                    } else {
                        log.error("shutDown Service offline error, active request still pending.");
                        return ServerResponse.status(503).body(Mono.just("Shutdown error. Active requests still pending."), String.class);
                    }
                }))
                .onErrorResume(e -> {
                    log.error("shutDown Service offline execution error.", e);
                    return ServerResponse.status(500).body(Mono.just("Shutdown error."), String.class);
                });
    }

    private boolean isNonLocalRequest(ServerRequest request) {
        if (request.remoteAddress().isEmpty()) {
            return true;
        }
        InetAddress remoteAddress = request.remoteAddress().get().getAddress();
        if (remoteAddress.isLoopbackAddress()) {
            return false;
        }
        return !remoteAddress.equals(localAddress);
    }

    private static Mono<ServerResponse> buildErrorResponse(String handler) {
        log.info("{} failed, remote request is not allowed.", handler);
        return ServerResponse.status(403).body(Mono.just("remote request is not allowed."), String.class);
    }
}
