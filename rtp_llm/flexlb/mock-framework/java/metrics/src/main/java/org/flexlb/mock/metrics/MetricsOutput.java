package org.flexlb.mock.metrics;

import io.prometheus.client.Collector.MetricFamilySamples;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Supplier;

/** Selects the event backend once; state and precomputed rate snapshots retain one periodic reader. */
public final class MetricsOutput implements AutoCloseable {
    private static final System.Logger LOG = System.getLogger(MetricsOutput.class.getName());
    private static final class Provider {
        static final MetricReporter INSTANCE = discover(ServiceLoader.load(MetricReporter.class), System.getenv());
    }
    private final ScheduledExecutorService worker;

    static MetricReporter eventReporter() { return Provider.INSTANCE; }

    static MetricReporter discover(Iterable<MetricReporter> providers, Map<String, String> environment) {
        var iterator = providers.iterator();
        if (!iterator.hasNext()) return null;
        var provider = Objects.requireNonNull(iterator.next());
        if (iterator.hasNext()) throw new IllegalStateException("Multiple Engine metric reporters installed");
        // Deliberately no fallback: a broken internal image must not silently lose monitoring.
        provider.initialize(Map.copyOf(environment));
        if (provider.intervalMillis() <= 0) throw new IllegalArgumentException("Invalid metric reporting interval");
        return provider;
    }

    public static MetricsOutput start(Supplier<List<MetricFamilySamples>> snapshot) {
        return new MetricsOutput(Provider.INSTANCE, snapshot);
    }

    MetricsOutput(MetricReporter reporter, Supplier<List<MetricFamilySamples>> snapshot) {
        Objects.requireNonNull(snapshot);
        if (reporter == null) { worker = null; return; }
        worker = Executors.newSingleThreadScheduledExecutor(Thread.ofPlatform().daemon(true).name("mock-metrics").factory());
        worker.scheduleWithFixedDelay(() -> {
            try { reporter.report(snapshot.get()); }
            catch (RuntimeException error) { LOG.log(System.Logger.Level.ERROR, "Engine metric reporting failed", error); }
        }, reporter.intervalMillis(), reporter.intervalMillis(), TimeUnit.MILLISECONDS);
    }

    public boolean prometheus() { return worker == null; }

    @Override public void close() {
        if (worker == null) return;
        worker.shutdownNow();
        try {
            if (!worker.awaitTermination(3, TimeUnit.SECONDS))
                throw new IllegalStateException("Engine metric reporter did not stop");
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted stopping Engine metric reporter", error);
        }
    }
}
