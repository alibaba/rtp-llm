package org.flexlb.mock.client;

import com.fasterxml.jackson.databind.*;
import com.sun.net.httpserver.HttpServer;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import org.flexlb.kvcm.grpc.*;
import java.net.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/** Native-protocol test fixture only: connects real Master queries to Engine event reports. */
public final class KvcmMetaProbe {
    public static void main(String[] args) throws Exception {
        var json = new ObjectMapper();
        var spaces = new ConcurrentHashMap<String, ConcurrentHashMap<String, Set<Long>>>();
        var queries = new AtomicLong(); var matchingHosts = new AtomicLong(); var rpcPort = new AtomicInteger(); var httpPort = new AtomicInteger();
        var ok = CommonResponseHeader.newBuilder().setStatus(org.flexlb.kvcm.grpc.Status.newBuilder().setCode(ErrorCode.OK)).build();
        var rpc = ServerBuilder.forPort(0).addService(new MetaServiceGrpc.MetaServiceImplBase() {
            @Override public void getClusterInfo(GetClusterInfoRequest request, StreamObserver<GetClusterInfoResponse> out) {
                out.onNext(GetClusterInfoResponse.newBuilder().setHeader(ok).setLeaderNodeId("leader").setLeaderEndpoint(
                    MetaNodeEndpoint.newBuilder().setNodeId("leader").setHost("127.0.0.1").setMetaRpcPort(rpcPort.get()).setMetaHttpPort(httpPort.get())).build()); out.onCompleted();
            }
            @Override public void getHostCacheState(GetHostCacheStateRequest request, StreamObserver<GetHostCacheStateResponse> out) {
                queries.incrementAndGet(); var response = GetHostCacheStateResponse.newBuilder().setHeader(ok);
                spaces.getOrDefault(request.getInstanceId(), new ConcurrentHashMap<>()).forEach((host, keys) -> {
                    int count = 0;
                    for (long key : request.getBlockCacheKeysList()) { if (!keys.contains(key)) break; count++; }
                    if (count > 0) matchingHosts.incrementAndGet();
                    response.addHosts(HostCacheMatch.newBuilder().setHostIpPort(host).setLocal(count));
                });
                out.onNext(response.build()); out.onCompleted();
            }
        }).build().start(); rpcPort.set(rpc.getPort());
        var http = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        http.createContext("/api/", exchange -> {
            try (exchange) {
                JsonNode body = json.readTree(exchange.getRequestBody());
                String instance = body.path("instance_id").asText();
                var hosts = spaces.computeIfAbsent(instance, key -> new ConcurrentHashMap<>());
                if (exchange.getRequestURI().getPath().equals("/api/reportEvent")) {
                    String host = body.path("host_ip_port").asText();
                    var keys = hosts.computeIfAbsent(host, key -> ConcurrentHashMap.newKeySet());
                    for (var event : body.path("events")) switch (event.path("event_type").asText()) {
                        case "EVENT_BLOCK_SNAPSHOT" -> {
                            var next = ConcurrentHashMap.<Long>newKeySet();
                            for (var block : event.path("block_snapshot").path("blocks")) next.add(Long.parseLong(block.path("block_key").asText()));
                            hosts.put(host, next);
                        }
                        case "EVENT_BLOCK_ADD" -> keys.add(Long.parseLong(event.path("block_add").path("block_key").asText()));
                        case "EVENT_BLOCK_DELETE" -> keys.remove(Long.parseLong(event.path("block_delete").path("block_key").asText()));
                        case "EVENT_HOST_DOWN" -> hosts.remove(host);
                    }
                }
                byte[] bytes = "{\"header\":{\"status\":{\"code\":1}}}".getBytes(java.nio.charset.StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(200, bytes.length); exchange.getResponseBody().write(bytes);
            }
        });
        http.createContext("/stats", exchange -> {
            try (exchange) {
                long keys = spaces.values().stream().flatMap(hosts -> hosts.values().stream()).mapToLong(Set::size).sum();
                byte[] bytes = json.writeValueAsBytes(Map.of("queries", queries.get(), "matching_hosts", matchingHosts.get(), "keys", keys));
                exchange.sendResponseHeaders(200, bytes.length); exchange.getResponseBody().write(bytes);
            }
        }); http.start(); httpPort.set(http.getAddress().getPort());
        Runtime.getRuntime().addShutdownHook(new Thread(() -> { http.stop(0); rpc.shutdownNow(); }));
        Path ready = Path.of(args[0]), temporary = ready.resolveSibling("kvcm-ready.tmp");
        json.writeValue(temporary.toFile(), Map.of("http", "http://127.0.0.1:" + http.getAddress().getPort(), "rpc_port", rpc.getPort()));
        Files.move(temporary, ready, StandardCopyOption.ATOMIC_MOVE);
        new CountDownLatch(1).await();
    }
}
