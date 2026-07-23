package org.flexlb.dispatcher;

import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceHostListener;
import org.flexlb.metric.NoOpFlexMonitor;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Predicate;
import java.util.function.Supplier;

import static org.flexlb.dispatcher.DispatcherMetricsReporter.CHUNK_OK;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Shared test scaffolding for the dispatcher subpackage. The dispatcher's domain beans take
 * "high-level" Spring beans in their production constructors (DispatcherFePoolRefresher,
 * FeHealthChecker, DispatchConfig). For unit tests that only care about a Supplier of URLs or
 * an isAlive predicate, these helpers wire up the minimum mocking needed so test bodies stay
 * focused on the behavior under test rather than the construction tax.
 */
final class DispatcherTestSupport {

    private DispatcherTestSupport() {
    }

    /**
     * A real {@link DispatcherMetricsReporter} backed by the no-op monitor — every report is a
     * cheap no-op, so tests that don't assert metrics can pass it without extra mocking.
     */
    static DispatcherMetricsReporter noopMetrics() {
        return new DispatcherMetricsReporter(NoOpFlexMonitor.getInstance());
    }

    /**
     * A {@link DispatcherMetricsReporter} that records every report invocation so tests can
     * assert on metric emission — how many times a method fired and with which tag values.
     * Backed by the no-op monitor, so nothing actually leaves the process.
     */
    static RecordingMetrics recordingMetrics() {
        return new RecordingMetrics();
    }

    static final class RecordingMetrics extends DispatcherMetricsReporter {

        record ChunkReport(String result, String reason, long rtMs) {
        }

        record RequestReport(String type, String path, int code, long costMs) {
        }

        /** Concurrent: reportChunk fires from Schedulers.parallel() workers during fanout. */
        final List<ChunkReport> chunkReports = new CopyOnWriteArrayList<>();
        final List<RequestReport> requestReports = new CopyOnWriteArrayList<>();

        RecordingMetrics() {
            super(NoOpFlexMonitor.getInstance());
        }

        @Override
        public void reportChunk(String reason, long rtMs) {
            chunkReports.add(new ChunkReport(CHUNK_OK.equals(reason) ? "ok" : "failed", reason, rtMs));
            super.reportChunk(reason, rtMs);
        }

        @Override
        public void reportRequest(String type, String path, int code, long costMs) {
            requestReports.add(new RequestReport(type, path, code, costMs));
            super.reportRequest(type, path, code, costMs);
        }
    }

    /**
     * Build a FePool whose snapshot is supplied by {@code urls} and whose liveness predicate is
     * {@code isAlive}. Wraps the production constructor by mocking
     * {@link DispatcherFePoolRefresher} and {@link FeHealthChecker}.
     */
    static FePool fePool(Supplier<List<String>> urls, Predicate<String> isAlive) {
        DispatcherFePoolRefresher refresher = mock(DispatcherFePoolRefresher.class);
        when(refresher.source()).thenReturn(urls);
        FeHealthChecker hc = mock(FeHealthChecker.class);
        when(hc.isAlive(anyString())).thenAnswer(inv -> isAlive.test(inv.getArgument(0)));
        return new FePool(refresher, hc);
    }

    /** Convenience overload for tests that don't care about per-host liveness. */
    static FePool fePool(List<String> staticUrls) {
        return fePool(() -> staticUrls, url -> true);
    }

    /**
     * Build a FeHealthChecker bound to {@code urls} as its probe source. Mocks
     * {@link DispatcherFePoolRefresher} and constructs a {@link DispatchConfig} POJO with
     * {@code probePath} set.
     */
    static FeHealthChecker feHealthChecker(Supplier<List<String>> urls, WebClient webClient, String probePath) {
        DispatcherFePoolRefresher refresher = mock(DispatcherFePoolRefresher.class);
        when(refresher.source()).thenReturn(urls);
        DispatchConfig cfg = new DispatchConfig();
        cfg.setProbePath(probePath);
        return new FeHealthChecker(refresher, webClient, cfg, noopMetrics());
    }

    /**
     * Build a DispatcherFePoolRefresher with an explicit serviceId. {@link DispatchConfig} is a
     * pure POJO, so we set the field directly. Grace stays at the unconfigured default
     * (non-positive → 5 minutes).
     */
    static DispatcherFePoolRefresher refresher(ServiceDiscovery sd, String serviceId) {
        return refresher(sd, serviceId, 0, System::nanoTime);
    }

    /** Overload with a controllable monotonic clock, for exercising the empty-discovery grace window. */
    static DispatcherFePoolRefresher refresher(ServiceDiscovery sd, String serviceId, java.util.function.LongSupplier nanoClock) {
        return refresher(sd, serviceId, 0, nanoClock);
    }

    /** Overload additionally configuring the empty-discovery grace, mirroring discoveryFailureGraceMs. */
    static DispatcherFePoolRefresher refresher(ServiceDiscovery sd, String serviceId,
            long emptyDiscoveryGraceMs, java.util.function.LongSupplier nanoClock) {
        DispatchConfig cfg = new DispatchConfig();
        cfg.setFePoolServiceId(serviceId);
        return new DispatcherFePoolRefresher(sd, cfg, emptyDiscoveryGraceMs, nanoClock);
    }

    /**
     * Test stub returning a controllable host list and exposing the registered listener so push
     * scenarios can be exercised without a real discovery client.
     */
    static final class StubServiceDiscovery implements ServiceDiscovery {
        private final String expectedId;
        private final AtomicReference<List<WorkerHost>> hosts;
        int getHostsCalls = 0;
        ServiceHostListener registeredListener = null;

        StubServiceDiscovery(String expectedId, WorkerHost... initial) {
            this.expectedId = expectedId;
            this.hosts = new AtomicReference<>(List.of(initial));
        }

        void setHosts(List<WorkerHost> updated) {
            this.hosts.set(updated);
        }

        void pushHosts(List<WorkerHost> updated) {
            this.hosts.set(updated);
            if (registeredListener != null) {
                registeredListener.onHostsChanged(updated);
            }
        }

        @Override
        public List<WorkerHost> getHosts(String address) {
            if (!expectedId.equals(address)) {
                throw new IllegalStateException("unexpected serviceId: " + address);
            }
            getHostsCalls++;
            return hosts.get();
        }

        @Override
        public void listen(String address, ServiceHostListener listener) {
            if (!expectedId.equals(address)) {
                throw new IllegalStateException("unexpected serviceId: " + address);
            }
            this.registeredListener = listener;
        }

        @Override
        public void shutdown() {}
    }
}
