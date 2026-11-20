package org.flexlb.mock.metrics;

import io.prometheus.client.Collector.MetricFamilySamples;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Supplier;

/**
 * Selects the event backend and schedules periodic state reporting.
 */
public final class MetricsOutput implements AutoCloseable {
    private static final System.Logger LOG = System.getLogger(MetricsOutput.class.getName());

    private final ScheduledExecutorService worker;
    private final Sender sender;

    MetricsOutput(MetricReporter reporter, Supplier<List<MetricFamilySamples>> snapshot) {
        Objects.requireNonNull(snapshot);
        if (reporter == null) { sender = null; worker = null; return; }
        sender = reporter instanceof Sender existing ? existing : new Sender(reporter, 65536);
        worker = Executors.newSingleThreadScheduledExecutor(Thread.ofPlatform().daemon(true).name("mock-metrics").factory());
        worker.scheduleWithFixedDelay(() -> {
            try { sender.checkHealthy(); sender.report(snapshot.get()); }
            catch (RuntimeException error) { LOG.log(System.Logger.Level.ERROR, "Engine metric reporting failed", error); }
        }, reporter.intervalMillis(), reporter.intervalMillis(), TimeUnit.MILLISECONDS);
    }

    public static MetricsOutput start(Supplier<List<MetricFamilySamples>> snapshot) {
        return new MetricsOutput(Provider.INSTANCE, snapshot);
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
        } finally { sender.close(); }
    }

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

    /**
     * Discovers the installed push backend and shares one sender across engine metrics.
     */
    private static final class Provider {
        static final MetricReporter RAW = discover(ServiceLoader.load(MetricReporter.class), System.getenv());
        static final Sender INSTANCE = RAW == null ? null : new Sender(RAW, 65536);
    }

    /**
     * Serializes SDK calls through a bounded queue so producers can continue without waiting.
     */
    static final class Sender implements MetricReporter, AutoCloseable {
        private final MetricReporter delegate;
        private final ThreadPoolExecutor executor;
        private final java.util.concurrent.atomic.AtomicReference<RuntimeException> failure = new java.util.concurrent.atomic.AtomicReference<>();

        Sender(MetricReporter delegate, int capacity) {
            this.delegate = Objects.requireNonNull(delegate);
            executor = new ThreadPoolExecutor(1, 1, 0, TimeUnit.SECONDS, new ArrayBlockingQueue<>(capacity),
                Thread.ofPlatform().daemon().name("mock-metric-sender").factory());
        }

        @Override public void initialize(Map<String, String> env) { throw new IllegalStateException("Reporter is already initialized"); }

        @Override public long intervalMillis() { return delegate.intervalMillis(); }

        @Override public void observe(String name, EventKind kind, Map<String, String> tags, double value) {
            var copiedTags = Map.copyOf(tags);
            send(() -> delegate.observe(name, kind, copiedTags, value));
        }

        @Override public void report(List<MetricFamilySamples> snapshot) {
            var copied = List.copyOf(snapshot);
            send(() -> delegate.report(copied));
        }

        @Override public void close() {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(3, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                    throw new IllegalStateException("Metric stream incomplete: sender did not drain");
                }
            } catch (InterruptedException error) {
                executor.shutdownNow(); Thread.currentThread().interrupt();
                throw new IllegalStateException("Interrupted draining metric sender", error);
            }
            checkHealthy();
        }

        void checkHealthy() {
            var error = failure.get();
            if (error != null) throw new IllegalStateException("Metric stream incomplete", error);
        }

        private void send(Runnable operation) {
            if (failure.get() != null) return;
            try {
                executor.execute(() -> {
                    try { operation.run(); }
                    catch (RuntimeException error) { failure.compareAndSet(null, error); }
                });
            } catch (RejectedExecutionException error) {
                failure.compareAndSet(null, new IllegalStateException("Metric stream incomplete: sender queue full or closed", error));
            }
        }
    }
}
