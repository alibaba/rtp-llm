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

import java.net.InetAddress;
import java.time.Duration;

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
            return buildErrorResponse();
        }
        if (processReady) {
            log.info("handleProcessOk success.");
            lifecycleReporter.reportProcessOk();
            return ServerResponse.ok().body(Mono.just("ok"), String.class);
        } else {
            return ServerResponse.status(503).body(Mono.just("not ready"), String.class);
        }
    }

    public Mono<ServerResponse> handleAppStartUp(ServerRequest request) {
        log.warn("recv /hook/after_start request.");
        if (isNonLocalRequest(request)) {
            return buildErrorResponse();
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
            return buildErrorResponse();
        }
        gracefulShutdownService.offline();
        log.info("shutDown Service run success, waiting for active request complete.");

        if (ActiveRequestShutdownHooker.shutdownCompletedSuccessfully) {
            log.info("shutDown Service offline success, active request complete.");
            return Mono.delay(Duration.ofMillis(3000))
                    .flatMap(v -> ServerResponse.ok().body(Mono.just("Shutdown complete. Ready for termination."), String.class));
        } else {
            log.error("shutDown Service offline error, active request still pending.");
            return Mono.delay(Duration.ofMillis(3000))
                    .flatMap(v -> ServerResponse.status(503).body(Mono.just("Shutdown error. Active requests still pending."), String.class));
        }
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

    private static Mono<ServerResponse> buildErrorResponse() {
        log.info("handleProcessOk failed, remote request is not allowed.");
        return ServerResponse.status(403).body(Mono.just("remote request is not allowed."), String.class);
    }
}
