package org.flexlb.balance.scheduler;

import com.google.common.collect.MapMaker;
import org.flexlb.balance.preemption.VictimResolution;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;

/** Exact-token execution barriers; weak identity keys prevent retained request history. */
public final class RequestSignals {
    private final ConcurrentMap<DeliveryClaim, CompletableFuture<Void>> deliveries = new MapMaker().weakKeys().makeMap();
    private final ConcurrentMap<PreemptionRegistration, CompletableFuture<VictimResolution>> preemptions = new MapMaker().weakKeys().makeMap();

    public CompletionStage<Void> delivery(DeliveryClaim claim) { return future(deliveries, claim).minimalCompletionStage(); }
    public void completeDelivery(DeliveryClaim claim) { future(deliveries, claim).complete(null); }
    public CompletionStage<VictimResolution> preemption(PreemptionRegistration claim) {
        return future(preemptions, claim).minimalCompletionStage();
    }
    public void resolvePreemption(PreemptionRegistration claim, VictimResolution result) {
        future(preemptions, claim).complete(result);
    }
    public VictimResolution resolved(PreemptionRegistration claim) {
        var future = preemptions.get(claim);
        return future == null ? null : future.getNow(null);
    }
    private static <K, V> CompletableFuture<V> future(ConcurrentMap<K, CompletableFuture<V>> map, K key) {
        var current = map.get(key);
        if (current != null) { return current; }
        var prepared = new CompletableFuture<V>();
        var raced = map.putIfAbsent(key, prepared);
        return raced == null ? prepared : raced;
    }
}
