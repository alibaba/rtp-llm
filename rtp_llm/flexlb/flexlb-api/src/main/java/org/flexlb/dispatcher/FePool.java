package org.flexlb.dispatcher;

import com.google.common.util.concurrent.RateLimiter;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.LongSupplier;

/** Owns FE membership, HTTP health and round-robin selection over immutable snapshots. */
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class FePool {
    private final ServiceDiscovery discovery;
    private final WebClient probeClient;
    private final DispatchConfig cfg;
    private final DispatcherMetricsReporter metrics;
    private final LongSupplier clock;
    private final long lookupTimeoutMs;
    private final long graceNanos;
    private final AtomicReference<List<String>> urls = new AtomicReference<>(List.of());
    private final ConcurrentHashMap<String, AtomicInteger> failures = new ConcurrentHashMap<>();
    private final AtomicLong cursor = new AtomicLong();
    private final AtomicBoolean allDeadReported = new AtomicBoolean();
    private final AtomicBoolean probing = new AtomicBoolean();
    private final RateLimiter emptyWarn = RateLimiter.create(1);
    private volatile long lastNonEmpty;
    // No queue: two interrupt-resistant lookups exhaust capacity without blocking Spring's timers.
    private final ExecutorService lookups = new ThreadPoolExecutor(0, 2, 60, TimeUnit.SECONDS,
            new SynchronousQueue<>(), Thread.ofPlatform().daemon().name("dispatcher-fe-discovery-", 0).factory());

    @Autowired
    public FePool(ServiceDiscovery discovery, @Qualifier("dispatcherProbeWebClient") WebClient probeClient,
                  DispatchConfig cfg, DispatcherMetricsReporter metrics) {
        this(discovery, probeClient, cfg, metrics, System::nanoTime, 3000);
    }

    FePool(ServiceDiscovery discovery, WebClient probeClient, DispatchConfig cfg,
           DispatcherMetricsReporter metrics, LongSupplier clock, long lookupTimeoutMs) {
        this.discovery = discovery;
        this.probeClient = probeClient;
        this.cfg = cfg;
        this.metrics = metrics;
        this.clock = clock;
        this.lookupTimeoutMs = lookupTimeoutMs;
        this.graceNanos = TimeUnit.MILLISECONDS.toNanos(
                cfg.getDiscoveryFailureGraceMs() > 0 ? cfg.getDiscoveryFailureGraceMs() : 300_000);
        this.lastNonEmpty = clock.getAsLong();
    }

    @PostConstruct
    public void start() {
        refresh();
        try {
            discovery.listen(cfg.getFePoolServiceId(), this::update);
        } catch (Exception error) {
            Logger.warn("FE discovery listener unavailable; polling continues: {}", error.toString());
        }
    }

    @Scheduled(fixedDelay = 30_000, initialDelay = 5_000)
    public void refresh() {
        Future<List<WorkerHost>> lookup = null;
        try {
            lookup = lookups.submit(() -> discovery.getHosts(cfg.getFePoolServiceId()));
            update(lookup.get(lookupTimeoutMs, TimeUnit.MILLISECONDS));
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
        } catch (Exception error) {
            Logger.warn("FE discovery refresh failed: {}", error.toString());
        } finally {
            if (lookup != null) {
                lookup.cancel(true);
            }
        }
    }

    void update(List<WorkerHost> hosts) {
        try {
            List<String> next = hosts.stream().map(host -> "http://" + host.getIpPort()).toList();
            if (!next.isEmpty()) {
                lastNonEmpty = clock.getAsLong();
                List<String> previous = urls.getAndSet(next);
                if (!previous.equals(next)) {
                    Logger.warn("FE pool updated: serviceId={}, size={} (was {})",
                            cfg.getFePoolServiceId(), next.size(), previous.size());
                }
            } else {
                List<String> previous = urls.get();
                if (!previous.isEmpty() && clock.getAsLong() - lastNonEmpty > graceNanos) {
                    if (urls.compareAndSet(previous, List.of())) {
                        Logger.warn("FE discovery remained empty beyond grace; dropping {} hosts", previous.size());
                    }
                } else if (!previous.isEmpty()) {
                    if (emptyWarn.tryAcquire()) {
                        Logger.warn("FE discovery empty within grace; retaining {} hosts", previous.size());
                    }
                }
            }
        } catch (RuntimeException error) {
            Logger.warn("Invalid FE discovery update; retaining previous snapshot: {}", error.toString());
        }
    }

    public int currentSize() {
        return urls.get().size();
    }

    boolean isAlive(String url) {
        AtomicInteger count = failures.get(url);
        return count == null || count.get() < 2;
    }

    @Scheduled(fixedRate = 1000)
    public void probeTick() {
        if (probing.compareAndSet(false, true)) {
            Mono.defer(this::probeOnce).doFinally(signal -> probing.set(false))
                    .subscribe(ignored -> { }, error -> Logger.warn("FE health round failed: {}", error.toString()));
        }
    }

    Mono<Void> probeOnce() {
        List<String> snapshot = urls.get();
        failures.keySet().retainAll(Set.copyOf(snapshot));
        metrics.reportFePool(snapshot.size(), (int) snapshot.stream().filter(this::isAlive).count());
        return Flux.fromIterable(snapshot).flatMap(url -> probeClient.get().uri(url + cfg.getProbePath())
                .retrieve().onStatus(status -> !status.is2xxSuccessful(), ClientResponse::createException)
                .toBodilessEntity().timeout(Duration.ofMillis(500))
                .doOnSuccess(response -> {
                    int previous = failures.computeIfAbsent(url, key -> new AtomicInteger()).getAndSet(0);
                    if (previous >= 2) {
                        Logger.warn("FE recovered: url={}, previousFailures={}", url, previous);
                    }
                })
                .onErrorResume(error -> {
                    int count = failures.computeIfAbsent(url, key -> new AtomicInteger()).incrementAndGet();
                    if (count == 2) {
                        Logger.warn("FE marked dead: url={}, err={}", url, error.getClass().getSimpleName());
                    }
                    return Mono.empty();
                })).then();
    }

    public String next() {
        List<String> pool = livePool();
        return pool.get(Math.floorMod(cursor.getAndIncrement(), pool.size()));
    }

    public List<String> nextBatch(int count) {
        if (count <= 0) {
            return List.of();
        }
        List<String> pool = livePool();
        long start = cursor.getAndAdd(count);
        List<String> picks = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            picks.add(pool.get(Math.floorMod(start + i, pool.size())));
        }
        return picks;
    }

    private List<String> livePool() {
        List<String> snapshot = urls.get();
        if (snapshot.isEmpty()) {
            throw new IllegalStateException("no FE endpoints available");
        }
        // Keep the common all-alive path allocation-free; copy only after the first dead host.
        List<String> alive = null;
        for (int i = 0; i < snapshot.size(); i++) {
            if (isAlive(snapshot.get(i))) {
                if (alive != null) {
                    alive.add(snapshot.get(i));
                }
            } else if (alive == null) {
                alive = new ArrayList<>(snapshot.subList(0, i));
            }
        }
        if (alive == null || !alive.isEmpty()) {
            allDeadReported.set(false);
            return alive == null ? snapshot : alive;
        }
        if (allDeadReported.compareAndSet(false, true)) {
            Logger.warn("FE pool all-dead fallback: size={}", snapshot.size());
        }
        return snapshot;
    }

    @PreDestroy
    void close() {
        lookups.shutdownNow();
    }
}
