package org.flexlb.service.grace;

import com.fasterxml.jackson.databind.JsonNode;
import org.flexlb.config.ConfigService;
import org.flexlb.config.NaviSchedPrefillSchedulerConfig;
import org.flexlb.config.PrefillSchedulerConfig;
import org.flexlb.util.JsonUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.event.ContextClosedEvent;
import org.springframework.context.event.EventListener;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;

@Component
public final class NaviSchedMasterRuntime implements AutoCloseable {
    @FunctionalInterface
    interface Launcher {
        Process start() throws IOException;
    }

    private final NaviSchedPrefillSchedulerConfig config;
    private final Launcher launcher;
    private final BooleanSupplier probe;
    private final Runnable onFailure;
    private ConfigurableApplicationContext owningContext;
    private Process child;
    private volatile boolean ready;
    private volatile boolean closing;
    private ScheduledExecutorService monitor;

    @Autowired
    public NaviSchedMasterRuntime(
            ConfigService configService,
            ConfigurableApplicationContext context,
            Environment environment) {
        this(
                configService.loadBalanceConfig().getPrefillScheduler()
                                instanceof NaviSchedPrefillSchedulerConfig navi
                        ? navi
                        : null,
                () -> startChild(configService, masterGrpcPort(environment)),
                () ->
                        probe(
                                configService.loadBalanceConfig().getPrefillScheduler()
                                                instanceof NaviSchedPrefillSchedulerConfig navi
                                        ? navi
                                        : null,
                                masterGrpcPort(environment)),
                context::close);
        owningContext = context;
        if (config != null) {
            masterGrpcPort(environment);
        }
    }

    NaviSchedMasterRuntime(
            NaviSchedPrefillSchedulerConfig config,
            Launcher launcher,
            BooleanSupplier probe,
            Runnable onFailure) {
        this.config = config;
        this.launcher = launcher;
        this.probe = probe;
        this.onFailure = onFailure;
    }

    static int masterGrpcPort(Environment environment) {
        int http = Integer.parseInt(environment.getProperty("server.port", "7001"));
        int grpc = http + 2 + 3;
        int management =
                Integer.parseInt(
                        environment.getProperty("management.server.port", String.valueOf(http)));
        if (http <= 0 || grpc > 65535) {
            throw new IllegalArgumentException("NaviSched derived port is out of range");
        }
        for (int port = grpc - 2; port <= grpc; port++) {
            if (port == management) {
                throw new IllegalArgumentException(
                        "NaviSched ports overlap FlexLB management port");
            }
        }
        return grpc;
    }

    private static Process startChild(ConfigService configService, int grpcPort)
            throws IOException {
        ProcessBuilder builder =
                new ProcessBuilder(
                                "python3", "-m", "navi_sched.adapters.rtp_llm.FlexlbPrefillMaster")
                        .inheritIO();
        builder.environment()
                .put(
                        "FLEXLB_CONFIG",
                        childConfiguration(
                                configService.loadBalanceConfig().getPrefillScheduler()));
        builder.environment().put("NAVI_SCHED_MASTER_HTTP_PORT", String.valueOf(grpcPort - 2));
        builder.environment().put("NAVI_SCHED_MASTER_ARPC_PORT", String.valueOf(grpcPort - 1));
        builder.environment().put("NAVI_SCHED_MASTER_GRPC_PORT", String.valueOf(grpcPort));
        return builder.start();
    }

    static String childConfiguration(PrefillSchedulerConfig scheduler) throws IOException {
        var document = JsonUtils.createObjectNode();
        document.put("schemaVersion", 3);
        document.set(
                "prefillScheduler", JsonUtils.readStrictTree(JsonUtils.toStrictString(scheduler)));
        return JsonUtils.toStrictString(document);
    }

    @PostConstruct
    public void start() throws IOException, InterruptedException {
        if (config == null) {
            ready = true;
            return;
        }
        child = launcher.start();
        try {
            long deadline =
                    System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(config.getStartupTimeoutMs());
            while (child.isAlive() && System.nanoTime() < deadline) {
                if (probe.getAsBoolean()) {
                    ready = true;
                    monitor =
                            Executors.newSingleThreadScheduledExecutor(
                                    runnable -> {
                                        Thread thread =
                                                new Thread(runnable, "navi-sched-master-monitor");
                                        thread.setDaemon(true);
                                        return thread;
                                    });
                    monitor.scheduleWithFixedDelay(this::checkHealth, 1, 1, TimeUnit.SECONDS);
                    return;
                }
                Thread.sleep(50);
            }
            throw new IllegalStateException("NaviSched Master did not become ready");
        } catch (RuntimeException | InterruptedException failure) {
            close();
            throw failure;
        }
    }

    void checkHealth() {
        if (closing || config == null) {
            return;
        }
        if (!child.isAlive() || !probe.getAsBoolean()) {
            ready = false;
            onFailure.run();
        }
    }

    public boolean isReady() {
        return ready && (config == null || child != null && child.isAlive());
    }

    private static boolean probe(NaviSchedPrefillSchedulerConfig config, int grpcPort) {
        if (config == null) {
            return true;
        }
        try (var client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(1)).build()) {
            var request =
                    HttpRequest.newBuilder(
                                    URI.create("http://127.0.0.1:" + (grpcPort - 2) + "/health"))
                            .timeout(Duration.ofSeconds(1))
                            .header("Content-Type", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString("{\"update\":false}"))
                            .build();
            var response = client.send(request, HttpResponse.BodyHandlers.ofString());
            JsonNode payload = JsonUtils.readStrictTree(response.body());
            return response.statusCode() == 200
                    && payload.path("accepted").asBoolean(false)
                    && payload.path("generation").asLong(0) > 0;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        } catch (Exception e) {
            return false;
        }
    }

    @EventListener(ContextClosedEvent.class)
    @Order(Ordered.LOWEST_PRECEDENCE)
    public void onContextClosed(ContextClosedEvent event) {
        if (event.getApplicationContext() == owningContext) {
            close();
        }
    }

    @Override
    public void close() {
        closing = true;
        ready = false;
        if (monitor != null) {
            monitor.shutdown();
        }
        if (child == null) {
            return;
        }
        child.destroy();
        try {
            if (!child.waitFor(config.getStopTimeoutMs(), TimeUnit.MILLISECONDS)) {
                child.destroyForcibly();
                if (!child.waitFor(config.getStopTimeoutMs(), TimeUnit.MILLISECONDS)) {
                    throw new IllegalStateException(
                            "NaviSched Master did not exit after forced stop");
                }
            }
        } catch (InterruptedException e) {
            child.destroyForcibly();
            Thread.currentThread().interrupt();
        }
    }
}
