package org.flexlb.mock.engine.vit;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Holds pending shared work and byte-budgeted embedding and hash LRU tiers.
 */
final class VitCache implements AutoCloseable {
    private final VitConfig config;
    private final VitMetrics metrics;
    private final Map<String, Pending> pending = new HashMap<>();
    private final LinkedHashMap<String, VitGeometry.Image> device = new LinkedHashMap<>(16,.75f,true), host = new LinkedHashMap<>(16,.75f,true), hashes = new LinkedHashMap<>(16,.75f,true);

    private long deviceBytes, hostBytes, hashBytes;
    private boolean closed;

    VitCache(VitConfig config, VitMetrics metrics) { this.config = config; this.metrics = metrics; }

    @Override public synchronized void close() {
        closed=true; device.clear(); host.clear(); hashes.clear(); deviceBytes=hostBytes=hashBytes=0;
        // Preparation owns terminal publication for outstanding entries.
    }

    synchronized List<Claim> acquire(List<String> keys) {
        if(closed) throw new IllegalStateException("V cache closed");
        var claims=new ArrayList<Claim>();
        try { for(var key:keys) claims.add(acquire(key)); return List.copyOf(claims); }
        catch(RuntimeException error) {
            // No newly created claim can escape this cache lock before commit.
            for(var claim:claims) {
                claim.pending.users.decrementAndGet();
                if(claim.producer) { pending.remove(claim.key); claim.pending.result.completeExceptionally(error); }
            }
            throw error;
        }
    }

    void publish(Claim claim, VitGeometry.Image image, Throwable error) {
        synchronized(this) {
            var current=pending.get(claim.key());
            if(current==null || current!=claim.pending) return;
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
            claim.pending.result.complete(image);
        } else {
            metrics.error(error,claim);
            claim.pending.result.completeExceptionally(error);
        }
    }

    long bytes(VitGeometry.Image image) { return image.bytes(config.hidden(),2); }

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

    private Claim acquire(String key) {
        if(closed) throw new IllegalStateException("V cache closed");
        var result=device.get(key);
        if(result==null && (result=host.remove(key))!=null) { hostBytes-=bytes(result); retain(result); }
        if(result!=null) {
            metrics.events.increment("py_rtp_vit_embedding_cache_hit_qps");
            return new Claim(key,new Pending(CompletableFuture.completedFuture(result),new AtomicInteger(1),
                new AtomicReference<>(),new AtomicBoolean()),false);
        }
        var waiting=pending.get(key);
        boolean producer=waiting==null;
        if(producer) {
            waiting=new Pending(new CompletableFuture<>(),new AtomicInteger(),new AtomicReference<>(),new AtomicBoolean());
            pending.put(key,waiting);
        }
        metrics.events.increment(producer?"py_rtp_vit_embedding_cache_miss_qps":"py_rtp_vit_embedding_cache_inflight_qps");
        waiting.users.incrementAndGet();
        return new Claim(key,waiting,producer);
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

    private static long hashBytes(VitGeometry.Image image) { return (long)image.rows()*4 + image.key().length(); }

    /**
     * Tracks one consumer of a shared image result and cancels work when the last consumer leaves.
     */
    static final class Claim implements AutoCloseable {
        private final String key;
        private final Pending pending;

        private final boolean producer;

        private final AtomicBoolean released = new AtomicBoolean();

        private Claim(String key, Pending pending, boolean producer) {
            this.key=key; this.pending=pending; this.producer=producer;
        }

        @Override public void close() {
            if(!released.compareAndSet(false,true)) return;
            int remaining=pending.users.decrementAndGet();
            var cancel=pending.abandoned.get(); if(remaining==0 && cancel!=null) cancel.run();
        }

        String key() { return key; }

        boolean producer() { return producer; }

        CompletionStage<VitGeometry.Image> result() { return pending.result.minimalCompletionStage(); }

        boolean markErrorReported() { return pending.errorReported.compareAndSet(false,true); }

        void onAbandon(Runnable action) { pending.abandoned.set(action); if(!hasConsumers()) action.run(); }

        boolean hasConsumers() { return pending.users.get()>0; }
    }

    private record Pending(CompletableFuture<VitGeometry.Image> result, AtomicInteger users,
                           AtomicReference<Runnable> abandoned, AtomicBoolean errorReported) {}
}
