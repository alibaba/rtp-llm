package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/** The frontend's VIP candidate + /master/info health and role selection for HA traffic. */
final class MasterRouteDiscovery implements AutoCloseable {
    record Host(String http, String grpc) {}
    record Info(String realMasterHost) {}
    record Route(String master, String slave) {}

    interface Candidates { List<Host> read() throws Exception; }
    interface Probe { Info read(Host host) throws Exception; }

    private static final class Health {
        Host host;
        boolean healthy;
        boolean master;
        int failures;
        long lastProbeMs;

        Health(Host host) { this.host = host; }
    }

    private final Candidates candidates;
    private final Probe probe;
    private final Map<String, Health> health = new HashMap<>();
    private List<Host> lastCandidates = List.of();
    private volatile Route route = new Route(null, null);
    private ScheduledExecutorService refreshThread;

    MasterRouteDiscovery(Candidates candidates, Probe probe) {
        this.candidates = candidates;
        this.probe = probe;
    }

    static MasterRouteDiscovery fromFile(Path path) {
        ObjectMapper mapper = new ObjectMapper();
        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(500)).build();
        Candidates candidates = () -> {
            JsonNode hosts = mapper.readTree(Files.readString(path)).path("hosts");
            if (!hosts.isArray()) {
                throw new IllegalArgumentException("master discovery requires a hosts array: " + path);
            }
            List<Host> result = new ArrayList<>();
            for (JsonNode row : hosts) {
                String http = row.path("http").asText("");
                String grpc = row.path("grpc").asText("");
                if (http.isBlank() || grpc.isBlank()) {
                    throw new IllegalArgumentException("master discovery host needs http and grpc: " + path);
                }
                result.add(new Host(http, grpc));
            }
            return result;
        };
        Probe probe = host -> {
            HttpRequest request = HttpRequest.newBuilder(
                    URI.create("http://" + host.http() + "/rtp_llm/master/info"))
                    .timeout(Duration.ofMillis(500))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString("{}"))
                    .build();
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                throw new IllegalStateException("master/info HTTP " + response.statusCode());
            }
            JsonNode body = mapper.readTree(response.body());
            return new Info(body.path("real_master_host").asText(""));
        };
        return new MasterRouteDiscovery(candidates, probe);
    }

    Route route() { return route; }

    synchronized void refresh() {
        try {
            lastCandidates = List.copyOf(candidates.read());
        } catch (Exception e) {
            // VipServer retains its last usable list on a transient refresh failure.
            System.err.println("master discovery refresh failed: " + e);
        }
        List<Host> toProbe = new ArrayList<>(lastCandidates);
        Set<String> seen = new HashSet<>();
        toProbe.removeIf(host -> !seen.add(host.http()));
        for (Health state : health.values()) {
            if (state.healthy && seen.add(state.host.http())) {
                toProbe.add(state.host);
            }
        }

        long now = System.currentTimeMillis();
        for (Host host : toProbe) {
            Health state = health.computeIfAbsent(host.http(), ignored -> new Health(host));
            state.host = host;
            try {
                Info info = probe.read(host);
                state.healthy = true;
                state.failures = 0;
                if (info.realMasterHost() != null && !info.realMasterHost().isBlank()) {
                    state.master = host.http().equals(info.realMasterHost());
                }
            } catch (Exception e) {
                state.failures++;
                if (state.failures >= 2) {
                    state.healthy = false;
                }
            }
            state.lastProbeMs = now;
        }
        health.values().removeIf(state -> !state.healthy
                && now - state.lastProbeMs > 30_000);

        List<Health> healthy = health.values().stream().filter(state -> state.healthy)
                .sorted(Comparator.comparing(state -> state.host.http())).toList();
        Health selected = healthy.stream().filter(state -> state.master).findFirst().orElse(null);
        if (selected == null && route.master() != null) {
            selected = healthy.stream().filter(state -> state.host.grpc().equals(route.master()))
                    .findFirst().orElse(null);
        }
        if (selected == null && !healthy.isEmpty()) {
            selected = healthy.get(0);
        }
        String master = selected == null ? null : selected.host.grpc();
        String slave = healthy.stream()
                .filter(state -> !state.master && !state.host.grpc().equals(master))
                .map(state -> state.host.grpc()).findFirst().orElse(null);
        Route next = new Route(master, slave);
        if (!next.equals(route)) {
            System.out.println("master route: master=" + master + ", slave=" + slave
                    + ", healthy=" + healthy.size());
        }
        route = next;
    }

    void start() {
        try {
            lastCandidates = List.copyOf(candidates.read());
        } catch (Exception e) {
            throw new IllegalArgumentException("cannot read initial master discovery", e);
        }
        if (lastCandidates.isEmpty()) {
            throw new IllegalArgumentException("initial master discovery has no hosts");
        }
        refresh();
        refreshThread = Executors.newSingleThreadScheduledExecutor(task -> {
            Thread thread = new Thread(task, "master-route-refresh");
            thread.setDaemon(true);
            return thread;
        });
        refreshThread.scheduleWithFixedDelay(() -> {
            try {
                refresh();
            } catch (RuntimeException e) {
                System.err.println("master route refresh failed: " + e);
            }
        }, 1, 1, TimeUnit.SECONDS);
    }

    @Override
    public void close() {
        if (refreshThread != null) {
            refreshThread.shutdownNow();
        }
    }
}
