package org.flexlb.mock.engine.vit;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/** Byte-budgeted embedding/hash LRU. Results held by callers outlive cache residency. */
final class VitCache implements AutoCloseable {
    record Claim(String key, CompletableFuture<VitGeometry.Image> result, AtomicInteger users, boolean producer,
                 AtomicReference<Runnable> abandoned, AtomicBoolean errorReported) {
        void onAbandon(Runnable action) { abandoned.set(action); if(users.get()==0) action.run(); }
    }
    private final VitConfig config;
    private final VitMetrics metrics;
    private record Pending(CompletableFuture<VitGeometry.Image> result, AtomicInteger users,
                           AtomicReference<Runnable> abandoned, AtomicBoolean errorReported) {}
    private final Map<String, Pending> pending = new HashMap<>();
    private final LinkedHashMap<String, VitGeometry.Image> device = new LinkedHashMap<>(16,.75f,true), host = new LinkedHashMap<>(16,.75f,true), hashes = new LinkedHashMap<>(16,.75f,true);
    private long deviceBytes, hostBytes, hashBytes;
    private boolean closed;
    VitCache(VitConfig config) { this(config, new VitMetrics()); }
    VitCache(VitConfig config, VitMetrics metrics) { this.config = config; this.metrics = metrics; }
    synchronized List<Claim> acquire(List<String> keys) {
        if(closed) throw new IllegalStateException("V cache closed");
        var claims=new ArrayList<Claim>();
        try { for(var key:keys) claims.add(acquire(key)); return List.copyOf(claims); }
        catch(RuntimeException error) {
            // No newly created claim can escape this cache lock before commit.
            for(var claim:claims) {
                claim.users.decrementAndGet();
                if(claim.producer) { pending.remove(claim.key); claim.result.completeExceptionally(error); }
            }
            throw error;
        }
    }
    private Claim acquire(String key) {
        if(closed) throw new IllegalStateException("V cache closed");
        var result=device.get(key);
        if(result==null && (result=host.remove(key))!=null) { hostBytes-=bytes(result); retain(result); }
        if(result!=null) {
            metrics.events.increment("py_rtp_vit_embedding_cache_hit_qps");
            return new Claim(key,CompletableFuture.completedFuture(result),new AtomicInteger(1),false,
                new AtomicReference<>(),new AtomicBoolean());
        }
        var waiting=pending.get(key);
        boolean producer=waiting==null;
        if(producer) {
            waiting=new Pending(new CompletableFuture<>(),new AtomicInteger(),new AtomicReference<>(),new AtomicBoolean());
            pending.put(key,waiting);
        }
        metrics.events.increment(producer?"py_rtp_vit_embedding_cache_miss_qps":"py_rtp_vit_embedding_cache_inflight_qps");
        waiting.users.incrementAndGet();
        return new Claim(key,waiting.result,waiting.users,producer,waiting.abandoned,waiting.errorReported);
    }
    void publish(Claim claim, VitGeometry.Image image, Throwable error) {
        synchronized(this) {
            var current=pending.get(claim.key());
            if(current==null || current.result!=claim.result) return;
            if(error==null && closed) error=io.grpc.Status.UNAVAILABLE.withDescription("V cache stopped").asRuntimeException();
            if(error==null) {
                try {
                    if(image==null || !image.key().equals(claim.key())) throw new IllegalArgumentException("V result identity mismatch");
                    retain(image);
                    long charge=hashBytes(image);
                    if(charge<=config.hashBytes()) {
                        var previous=hashes.remove(claim.key()); if(previous!=null) hashBytes-=hashBytes(previous);
                        while(hashBytes+charge>config.hashBytes() && !hashes.isEmpty()) {
                            var old=hashes.pollFirstEntry(); hashBytes-=hashBytes(old.getValue());
                        }
                        hashes.put(claim.key(),image); hashBytes+=charge;
                    }
                } catch(RuntimeException failure) {
                    error=failure;
                    var gpu=device.remove(claim.key()); if(gpu!=null) deviceBytes-=bytes(gpu);
                    var cpu=host.remove(claim.key()); if(cpu!=null) hostBytes-=bytes(cpu);
                    var hash=hashes.remove(claim.key()); if(hash!=null) hashBytes-=hashBytes(hash);
                }
            }
            pending.remove(claim.key());
        }
        // Storage and metadata are visible before consumers observe the terminal result.
        // Never execute consumer callbacks under the cache lock.
        if(error==null) {
            metrics.events.increment("py_rtp_vit_success_qps");
            claim.result.complete(image);
        } else {
            metrics.error(error,claim);
            claim.result.completeExceptionally(error);
        }
    }
    private void retain(VitGeometry.Image image) {
        if (device.containsKey(image.key()) || host.containsKey(image.key())) return;
        long bytes = bytes(image);
        if (bytes <= config.deviceBytes() && config.deviceBytes() > 0) {
            while (deviceBytes > config.deviceBytes() - bytes) {
                var old = device.pollFirstEntry(); deviceBytes -= bytes(old.getValue()); storeHost(old.getValue(),true);
            }
            device.put(image.key(),image); deviceBytes += bytes;
        } else storeHost(image,false);
    }
    private void storeHost(VitGeometry.Image image, boolean evictedFromDevice) {
        long bytes = bytes(image);
        if (bytes > config.hostBytes() || config.hostBytes() == 0) {
            if(evictedFromDevice) metrics.events.increment("py_rtp_vit_embedding_cache_eviction_qps");
            return;
        }
        while (hostBytes > config.hostBytes() - bytes) { var old = host.pollFirstEntry(); hostBytes -= bytes(old.getValue()); metrics.events.increment("py_rtp_vit_embedding_cache_eviction_qps"); }
        host.put(image.key(),image); hostBytes += bytes;
    }
    static void release(Claim claim) {
        int remaining=claim.users.decrementAndGet();
        if(remaining<0) throw new IllegalStateException("V claim released twice");
        var cancel=claim.abandoned.get(); if(remaining==0 && cancel!=null) cancel.run();
    }
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
    @Override public synchronized void close() {
        closed=true; device.clear(); host.clear(); hashes.clear(); deviceBytes=hostBytes=hashBytes=0;
        // Preparation owns terminal publication for outstanding entries.
    }
}
