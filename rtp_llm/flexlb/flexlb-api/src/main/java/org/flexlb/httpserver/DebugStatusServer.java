package org.flexlb.httpserver;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.debug.DebugQuery;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;

import javax.annotation.PreDestroy;
import java.io.ByteArrayOutputStream;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/** Opt-in application-port diagnostics, with one worker and one waiting capture. */
@Component
@ConditionalOnProperty(name = "flexlb.debug.enabled", havingValue = "true")
public class DebugStatusServer {
    private static final int MAX_RESPONSE_BYTES = 8 * 1024 * 1024;
    private final DebugSnapshotService service;
    private final ObjectMapper mapper;
    private final ThreadPoolExecutor executor = new ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS,
            new ArrayBlockingQueue<>(1), runnable -> {
                Thread thread = new Thread(runnable, "flexlb-debug-snapshot");
                thread.setDaemon(true);
                return thread;
            }, new ThreadPoolExecutor.AbortPolicy());
    private final Scheduler worker = Schedulers.fromExecutorService(executor);

    public DebugStatusServer(DebugSnapshotService service, ObjectMapper mapper) {
        this.service = service;
        this.mapper = mapper;
    }

    @Bean
    public RouterFunction<ServerResponse> debugRoutes() {
        return RouterFunctions.route()
                .GET("/rtp_llm/debug/snapshot", request -> handle(request, false))
                .GET("/rtp_llm/debug/requests/{id}", request -> handle(request, true))
                .build();
    }

    private Mono<ServerResponse> handle(ServerRequest request, boolean single) {
        final DebugQuery query;
        final Set<String> include;
        final int endpointLimit;
        try {
            Long id = single ? Long.valueOf(request.pathVariable("id")) : null;
            int limit = Integer.parseInt(request.queryParam("limit").orElse("500"));
            int scans = Integer.parseInt(request.queryParam("scan_limit").orElse("2000"));
            query = new DebugQuery(limit, scans, id);
            include = Arrays.stream(request.queryParam("include")
                    .orElse("scheduler,queues,prefill,decode,engine").split(",", -1))
                    .collect(Collectors.toUnmodifiableSet());
            endpointLimit = Integer.parseInt(request.queryParam("endpoint_limit").orElse("64"));
            if (include.isEmpty() || !DebugSnapshotService.INCLUDES.containsAll(include)
                    || endpointLimit < 1 || endpointLimit > 256) {
                throw new IllegalArgumentException("invalid include or endpoint_limit (1..256)");
            }
        } catch (IllegalArgumentException invalid) {
            return error(400, "invalid_query");
        }
        return Mono.fromCallable(() -> {
                    BoundedOutput output = new BoundedOutput();
                    mapper.writeValue(output, service.capture(query, include, endpointLimit));
                    return output.toByteArray();
                }).subscribeOn(worker)
                .flatMap(bytes -> ServerResponse.ok().contentType(MediaType.APPLICATION_JSON)
                        .header("Cache-Control", "no-store").bodyValue(bytes))
                // Failed/rejected/oversized captures never masquerade as empty snapshots.
                .onErrorResume(failure -> error(503, "capture_unavailable"));
    }

    private static Mono<ServerResponse> error(int status, String code) {
        return ServerResponse.status(status).contentType(MediaType.APPLICATION_JSON)
                .header("Cache-Control", "no-store").bodyValue(Map.of("error", code));
    }

    @PreDestroy
    public void close() {
        worker.dispose();
        executor.shutdownNow();
    }

    private static final class BoundedOutput extends ByteArrayOutputStream {
        private void check(int length) {
            if (length > MAX_RESPONSE_BYTES - count) {
                throw new IllegalStateException("debug response exceeds byte budget");
            }
        }

        @Override
        public synchronized void write(int value) {
            check(1);
            super.write(value);
        }

        @Override
        public synchronized void write(byte[] bytes, int offset, int length) {
            check(length);
            super.write(bytes, offset, length);
        }
    }
}
