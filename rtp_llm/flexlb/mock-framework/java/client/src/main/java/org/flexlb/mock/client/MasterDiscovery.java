package org.flexlb.mock.client;

import com.fasterxml.jackson.databind.*;
import java.io.*;
import java.net.*;
import java.net.http.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;

/** Mirrors Frontend MasterService heartbeat selection; topology.json replaces VIP discovery. */
final class MasterDiscovery implements AutoCloseable {
    private static final ObjectMapper JSON = new ObjectMapper();
    private final Path topology;
    private final Set<String> allowed;
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofMillis(500)).build();
    private final ExecutorService probes = Executors.newFixedThreadPool(4);
    private final ScheduledExecutorService refresh = Executors.newSingleThreadScheduledExecutor();
    private final Map<String, Host> hosts = new HashMap<>();
    private volatile List<String> routes = List.of();
    private volatile Exception failure;
    private String primary;

    private static final class Host {
        final String http, rpc;
        boolean healthy, leader;
        int failures;
        long lastProbe;
        Host(String http, String rpc) { this.http = http; this.rpc = rpc; }
    }
    MasterDiscovery(Path topology, List<String> allowed) throws Exception {
        this.topology = topology;
        this.allowed = allowed == null ? null : Set.copyOf(allowed);
        try {
            update();
            refresh.scheduleAtFixedRate(() -> {
                try { update(); }
                catch (Exception error) { failure = error; routes = List.of(); }
            }, 1, 1, TimeUnit.SECONDS);
        } catch (Exception error) { close(); throw error; }
    }
    void check() {
        if (failure != null) throw new IllegalStateException("Master discovery failed", failure);
    }
    List<String> routes() { check(); return routes; }

    private JsonNode heartbeat(Host host) {
        try {
            var request = HttpRequest.newBuilder(URI.create(host.http + "/rtp_llm/master/info"))
                .timeout(Duration.ofMillis(500)).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString("{}")).build();
            var response = http.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() == 200) {
                var value = JSON.readTree(response.body());
                if (value != null && value.isObject()) return value;
            }
        } catch (IOException | IllegalArgumentException error) {
            // Failed heartbeats are availability observations, not playback infrastructure errors.
        } catch (InterruptedException error) { Thread.currentThread().interrupt(); }
        return null;
    }
    private void update() throws Exception {
        var nodes = JSON.readTree(topology.toFile());
        if (nodes == null || !nodes.isArray()) throw new IOException("Invalid topology array");
        var discovered = new LinkedHashSet<String>();
        for (var node : nodes) {
            if (!node.path("role").asText().equals("master")) continue;
            String rpc = node.path("rpc").asText(), address = node.path("http").asText();
            var uri = URI.create(address);
            if (!"http".equals(uri.getScheme()) || uri.getHost() == null || uri.getPort() < 1
                    || !rpc.equals(uri.getHost() + ":" + (uri.getPort() + 2)))
                throw new IOException("Invalid native Master endpoints: " + node);
            if (allowed != null && !allowed.contains(rpc)) continue;
            if (!discovered.add(rpc)) throw new IOException("Duplicate Master endpoint");
            hosts.computeIfAbsent(rpc, k -> new Host(address, rpc));
        }
        // Frontend continues probing cached healthy hosts when discovery temporarily omits them.
        hosts.values().stream().filter(h -> h.healthy).forEach(h -> discovered.add(h.rpc));
        var pending = new LinkedHashMap<Host, Future<JsonNode>>();
        for (String rpc : discovered) {
            Host host = hosts.get(rpc);
            pending.put(host, probes.submit(() -> heartbeat(host)));
        }
        for (var entry : pending.entrySet()) {
            Host host = entry.getKey();
            JsonNode value = entry.getValue().get();
            host.lastProbe = System.nanoTime();
            if (value == null) {
                if (++host.failures >= 2) host.healthy = false;
            } else {
                host.healthy = true; host.failures = 0;
                String leader = value.path("real_master_host").asText();
                if (!leader.isEmpty()) host.leader = leader.equals(URI.create(host.http).getAuthority());
            }
        }
        hosts.values().removeIf(h -> !h.healthy && System.nanoTime() - h.lastProbe > TimeUnit.SECONDS.toNanos(30));
        var healthy = hosts.values().stream().filter(h -> h.healthy)
            .sorted(Comparator.comparing(h -> URI.create(h.http).getHost())).toList();
        Host selected = healthy.stream().filter(h -> h.leader).findFirst().orElse(null);
        if (selected == null && primary != null) {
            Host previous = hosts.get(primary);
            if (previous != null && previous.healthy) selected = previous;
        }
        if (selected == null && !healthy.isEmpty()) selected = healthy.getFirst();
        if (selected == null) { primary = null; routes = List.of(); return; }
        primary = selected.rpc;
        String chosen = primary;
        Host backup = healthy.stream().filter(h -> !h.leader && !h.rpc.equals(chosen)).findFirst().orElse(null);
        routes = backup == null ? List.of(primary) : List.of(primary, backup.rpc);
    }
    @Override public void close() {
        refresh.shutdownNow(); probes.shutdownNow(); http.shutdownNow();
        try { refresh.awaitTermination(2, TimeUnit.SECONDS); probes.awaitTermination(2, TimeUnit.SECONDS); }
        catch (InterruptedException error) { Thread.currentThread().interrupt(); }
    }
}
