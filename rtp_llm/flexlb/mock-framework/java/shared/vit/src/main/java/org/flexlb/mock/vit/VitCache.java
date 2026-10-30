package org.flexlb.mock.vit;

import org.flexlb.mock.multimodal.MediaInput;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/** Byte-budgeted embedding/hash LRU. Results held by callers outlive cache residency. */
final class VitCache implements AutoCloseable {
    record Claim(VitGeometry.Image image, CompletableFuture<VitGeometry.Image> result, AtomicInteger users, boolean producer) {}
    private final VitConfig config;
    private final VitMetrics metrics;
    private record Pending(CompletableFuture<VitGeometry.Image> result, AtomicInteger users) {}
    private final Map<String, Pending> pending = new HashMap<>();
    private final LinkedHashMap<String, VitGeometry.Image> device = new LinkedHashMap<>(16,.75f,true), host = new LinkedHashMap<>(16,.75f,true), hashes = new LinkedHashMap<>(16,.75f,true);
    private long deviceBytes, hostBytes, hashBytes;
    private boolean closed;
    VitCache(VitConfig config) { this(config, new VitMetrics()); }
    VitCache(VitConfig config, VitMetrics metrics) { this.config = config; this.metrics = metrics; }
    synchronized Claim acquire(VitGeometry.Image image) {
        if (closed) throw new IllegalStateException("V cache closed");
        var result = device.get(image.key());
        if (result == null && (result = host.remove(image.key())) != null) {
            hostBytes -= bytes(result); retain(result);
        }
        if (result != null) { metrics.events.increment("py_rtp_vit_embedding_cache_hit_qps"); return new Claim(image, CompletableFuture.completedFuture(result),new AtomicInteger(1),false); }
        var waiting = pending.get(image.key());
        if (waiting != null) { metrics.events.increment("py_rtp_vit_embedding_cache_inflight_qps"); waiting.users.incrementAndGet();return new Claim(image,waiting.result,waiting.users,false); }
        metrics.events.increment("py_rtp_vit_embedding_cache_miss_qps"); waiting = new Pending(new CompletableFuture<>(),new AtomicInteger(1));pending.put(image.key(),waiting);
        return new Claim(image,waiting.result,waiting.users,true);
    }
    void complete(Claim claim, Throwable error) {
        synchronized (this) {
            var current=pending.get(claim.image.key());
            if(current==null || current.result!=claim.result) return;
            pending.remove(claim.image.key());
            if (error == null && !closed) {
                retain(claim.image);
                long charge = hashBytes(claim.image);
                if (charge <= config.hashBytes()) {
                    var previous = hashes.remove(claim.image.key()); if (previous != null) hashBytes -= hashBytes(previous);
                    while (hashBytes + charge > config.hashBytes() && !hashes.isEmpty()) {
                        var old = hashes.pollFirstEntry(); hashBytes -= hashBytes(old.getValue());
                    }
                    hashes.put(claim.image.key(),claim.image); hashBytes += charge;
                }
            }
        }
        // Completion callbacks may acquire scheduler/RPC locks; never run them under the cache lock.
        if (error == null) claim.result.complete(claim.image); else claim.result.completeExceptionally(error);
    }
    private void retain(VitGeometry.Image image) {
        if (device.containsKey(image.key()) || host.containsKey(image.key())) return;
        long bytes = bytes(image);
        if (bytes <= config.deviceBytes() && config.deviceBytes() > 0) {
            while (deviceBytes > config.deviceBytes() - bytes) {
                var old = device.pollFirstEntry(); deviceBytes -= bytes(old.getValue()); storeHost(old.getValue());
            }
            device.put(image.key(),image); deviceBytes += bytes;
        } else storeHost(image);
    }
    private void storeHost(VitGeometry.Image image) {
        long bytes = bytes(image);
        if (bytes > config.hostBytes() || config.hostBytes() == 0) return;
        while (hostBytes > config.hostBytes() - bytes) { var old = host.pollFirstEntry(); hostBytes -= bytes(old.getValue()); metrics.events.increment("py_rtp_vit_embedding_cache_eviction_qps"); }
        host.put(image.key(),image); hostBytes += bytes;
    }
    static void release(Claim claim) { if(claim.users.decrementAndGet()<0) throw new IllegalStateException("V claim released twice"); }
    long bytes(VitGeometry.Image image) { return image.bytes(config.hidden(),2); }
    private static long hashBytes(VitGeometry.Image image) { return (long)image.rows()*4 + image.key().length(); }
    synchronized VitGeometry.Image hash(String key) { return hashes.get(key); }
    synchronized String tier(String key) { return device.containsKey(key) ? "gpu" : host.containsKey(key) ? "cpu" : ""; }
    synchronized Set<String> keys() { return Set.copyOf(hashes.keySet()); }
    synchronized Set<String> deviceKeys() { return Set.copyOf(device.keySet()); }
    synchronized Set<String> hostKeys() { return Set.copyOf(host.keySet()); }
    synchronized Map<String, Double> snapshot() {
        long tokens = (deviceBytes + hostBytes) / (2L * config.hidden());
        return Map.of("py_rtp_vit_embedding_cache_tokens", (double) tokens,
            "py_rtp_vit_embedding_cache_bytes", (double) (deviceBytes + hostBytes),
            "mock_vit_hash_cache_bytes", (double) hashBytes);
    }
    @Override public void close() {
        List<CompletableFuture<VitGeometry.Image>> waiting;
        synchronized (this) {
            if (closed) return; closed=true;
            waiting=pending.values().stream().map(Pending::result).toList(); pending.clear(); device.clear();host.clear();hashes.clear();deviceBytes=hostBytes=hashBytes=0;
        }
        waiting.forEach(f -> f.completeExceptionally(io.grpc.Status.UNAVAILABLE.withDescription("V stopped").asRuntimeException()));
    }
}
