package org.flexlb.dispatcher;

import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

/**
 * Tracks FE-host liveness via periodic HTTP probes of an application-level path (default
 * {@code /frontend_health} for rtp_llm; switch via {@code DISPATCH_PROBE_PATH} for vLLM
 * ({@code /health}) or other backends). The probe targets a FE-only endpoint that bypasses
 * backend health, so the dispatcher's view of FE doesn't get poisoned when BE is down. Design
 * parameters mirror {@code host_service.py}'s production-tuned {@code MasterService} so
 * cross-language ops behavior stays consistent.
 *
 * <p>Why bother when VipServer already drops unhealthy nodes from the registry: VipServer's
 * probe only proves TCP reachability. An OOM'd / event-loop-stuck FE keeps its port open but
 * stops responding — only an application-level probe catches that. VipServer is layer 1
 * (registration), this is layer 2 (application liveness).
 */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class FeHealthChecker {

    private static final int FAIL_THRESHOLD = 2;
    private static final int PROBE_TIMEOUT_MS = 500;
    private static final long PROBE_INTERVAL_MS = 1000;

    private final Supplier<List<String>> urlSupplier;
    private final WebClient webClient;
    private final String probePath;
    private final ConcurrentMap<String, AtomicInteger> consecFails = new ConcurrentHashMap<>();
    private ScheduledExecutorService scheduler;

    public FeHealthChecker(DispatcherFePoolRefresher refresher,
                           @Qualifier("dispatcherPassthroughWebClient") WebClient webClient,
                           DispatchConfig cfg) {
        String probePath = cfg.getProbePath();
        if (probePath == null || probePath.isBlank()) {
            throw new IllegalArgumentException(
                    "probePath must not be blank — pass /frontend_health, /health, etc.");
        }
        this.urlSupplier = refresher.source();
        this.webClient = webClient;
        this.probePath = probePath;
    }

    /**
     * Optimistic default: an unprobed URL is assumed alive. Removing a host from the pool the
     * instant it appears (before its first probe completes) would block legitimate traffic on a
     * data race; the opposite — letting one in-flight request hit a freshly-dead host before the
     * next probe — is recoverable at the caller.
     */
    public boolean isAlive(String url) {
        AtomicInteger n = consecFails.get(url);
        return n == null || n.get() < FAIL_THRESHOLD;
    }

    /**
     * Raw consecutive-failure counter for {@code url}. {@code 0} for never-probed or
     * currently-healthy; {@code 1} after the first failure (still alive, in flap-tolerance
     * window); {@code >= FAIL_THRESHOLD} once {@link #isAlive(String)} flips to false. Exposed
     * for the snapshot endpoint so operators can distinguish "warming up to dead" from
     * "freshly dead".
     */
    public int consecFails(String url) {
        AtomicInteger n = consecFails.get(url);
        return n == null ? 0 : n.get();
    }

    /**
     * Run one probe round against the current snapshot of {@link #urlSupplier}. Each URL gets a
     * single {@code GET <url><probePath>} with {@value #PROBE_TIMEOUT_MS}ms timeout; 2xx resets
     * the failure counter, everything else (non-2xx, connect refused, read timeout) increments it.
     * Probes run in parallel via reactor; the returned {@code Mono} completes when all are done.
     */
    public Mono<Void> probeOnce() {
        List<String> urls = urlSupplier.get();
        if (urls == null || urls.isEmpty()) {
            return Mono.empty();
        }
        return Flux.fromIterable(urls)
                .flatMap(url -> webClient.get()
                        .uri(url + probePath)
                        .retrieve()
                        .toBodilessEntity()
                        .timeout(Duration.ofMillis(PROBE_TIMEOUT_MS))
                        .doOnSuccess(r -> {
                            AtomicInteger counter = consecFails
                                    .computeIfAbsent(url, k -> new AtomicInteger());
                            int prev = counter.getAndSet(0);
                            // Only log on dead→alive transition. Every-probe success would
                            // make pv.log unreadable and adds nothing once the host is known
                            // healthy.
                            if (prev >= FAIL_THRESHOLD) {
                                Logger.warn("FE recovered: url={}, after {} consec failures",
                                        url, prev);
                            }
                        })
                        .onErrorResume(e -> {
                            int n = consecFails
                                    .computeIfAbsent(url, k -> new AtomicInteger())
                                    .incrementAndGet();
                            // Log the exact tick the host crosses the threshold, once. Every
                            // failure spam would flood the log under a real outage; once-on-
                            // transition tells ops the minute it happened and the cause.
                            if (n == FAIL_THRESHOLD) {
                                Logger.warn("FE marked dead: url={}, consec={}, err={}",
                                        url, n, e.getClass().getSimpleName());
                            } else {
                                Logger.debug("FE probe failed: url={}, consec={}, err={}",
                                        url, n, e.getClass().getSimpleName());
                            }
                            return Mono.empty();
                        })
                        .then())
                .then();
    }

    /**
     * Start the background probe loop. Idempotent — subsequent calls are no-ops once started.
     */
    public synchronized void start() {
        if (scheduler != null) {
            return;
        }
        scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "fe-health-checker");
            t.setDaemon(true);
            return t;
        });
        // Catch synchronous throws so a future urlSupplier that can throw doesn't make
        // ScheduledExecutorService silently cancel the task — once the loop dies all hosts
        // stay optimistically isAlive=true and dead-host filtering quietly stops working.
        scheduler.scheduleAtFixedRate(() -> {
            try {
                probeOnce().subscribe();
            } catch (Throwable t) {
                Logger.warn("FE health probe round threw, scheduler kept alive: err={}: {}",
                        t.getClass().getSimpleName(), t.getMessage());
            }
        }, 0, PROBE_INTERVAL_MS, TimeUnit.MILLISECONDS);
    }

    public synchronized void stop() {
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler = null;
        }
    }

    @PostConstruct
    void startProbes() {
        start();
    }

    @PreDestroy
    void stopProbes() {
        stop();
    }
}
