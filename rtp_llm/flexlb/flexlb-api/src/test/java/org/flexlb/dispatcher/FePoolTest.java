package org.flexlb.dispatcher;

import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceHostListener;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.mockito.ArgumentCaptor;
import org.springframework.http.HttpStatus;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@Timeout(10)
class FePoolTest {
    private final ServiceDiscovery discovery = mock(ServiceDiscovery.class);
    private final DispatchConfig cfg = new DispatchConfig();
    private final AtomicLong clock = new AtomicLong();
    private FePool pool;

    @AfterEach
    void close() {
        if (pool != null) {
            pool.close();
        }
    }

    private FePool create(WebClient client) {
        cfg.setDiscoveryFailureGraceMs(100);
        pool = new FePool(discovery, client, cfg, DispatcherTestSupport.noopMetrics(), clock::get, 50);
        return pool;
    }

    @Test
    void discoveryPushPollGraceAndRecoveryPublishCompleteSnapshots() {
        when(discovery.getHosts(anyString())).thenThrow(new IllegalStateException("boot unavailable"));
        create(WebClient.create()).start();
        assertThrows(IllegalStateException.class, pool::next);
        ArgumentCaptor<ServiceHostListener> listener = ArgumentCaptor.forClass(ServiceHostListener.class);
        verify(discovery).listen(anyString(), listener.capture());
        listener.getValue().onHostsChanged(List.of(WorkerHost.of("a", 80), WorkerHost.of("b", 80)));
        assertEquals(List.of("http://a:80", "http://b:80", "http://a:80"), pool.nextBatch(3));
        assertEquals("http://b:80", pool.next());
        doReturn(List.of()).when(discovery).getHosts(anyString());
        pool.refresh();
        assertEquals(2, pool.currentSize());
        clock.set(TimeUnit.MILLISECONDS.toNanos(101));
        pool.refresh();
        assertThrows(IllegalStateException.class, pool::next);
        listener.getValue().onHostsChanged(List.of(WorkerHost.of("new", 80)));
        assertEquals("http://new:80", pool.next());
        listener.getValue().onHostsChanged(null);
        assertEquals("http://new:80", pool.next());
    }

    @Test
    void probesFilterDeadHostsRecoverAndDoNotOverlap() {
        AtomicInteger status = new AtomicInteger(503);
        AtomicInteger probes = new AtomicInteger();
        Sinks.One<ClientResponse> slow = Sinks.one();
        create(WebClient.builder().exchangeFunction(request -> {
            probes.incrementAndGet();
            assertEquals("/frontend_health", request.url().getPath());
            if (status.get() == 0) {
                return slow.asMono();
            }
            int code = request.url().getHost().equals("a") ? status.get() : 200;
            return Mono.just(ClientResponse.create(HttpStatus.valueOf(code)).build());
        }).build());
        pool.update(List.of(WorkerHost.of("a", 80), WorkerHost.of("b", 80)));
        pool.probeOnce().block();
        assertTrue(pool.isAlive("http://a:80"));
        pool.probeOnce().block();
        assertEquals(List.of("http://b:80", "http://b:80"), pool.nextBatch(2));
        status.set(200);
        pool.probeOnce().block();
        assertTrue(pool.isAlive("http://a:80"));
        status.set(302);
        pool.probeOnce().block();
        pool.probeOnce().block();
        pool.update(List.of(WorkerHost.of("a", 80)));
        assertEquals("http://a:80", pool.next()); // all-dead fallback
        status.set(0);
        pool.probeTick();
        int started = probes.get();
        pool.probeTick();
        assertEquals(started, probes.get());
        slow.tryEmitValue(ClientResponse.create(HttpStatus.OK).build());
        pool.probeTick();
        assertEquals(started + 1, probes.get());
        clock.set(TimeUnit.SECONDS.toNanos(1));
        pool.update(List.of());
        pool.probeOnce().block();
        assertTrue(pool.isAlive("http://a:80"));
    }

    @Test
    void concurrentReservationsStayBalanced() {
        create(WebClient.create()).update(List.of(WorkerHost.of("a", 80), WorkerHost.of("b", 80), WorkerHost.of("c", 80)));
        List<String> picks = IntStream.range(0, 900).parallel().mapToObj(i -> pool.nextBatch(7)).flatMap(List::stream).toList();
        for (String host : List.of("a", "b", "c")) {
            assertEquals(2100, picks.stream().filter(("http://" + host + ":80")::equals).count());
        }
    }

    @Test
    void stuckDiscoveryLookupsAreBoundedAndCapacityRecovers() throws Exception {
        CountDownLatch release = new CountDownLatch(1);
        AtomicInteger calls = new AtomicInteger();
        when(discovery.getHosts(anyString())).thenAnswer(ignored -> {
            if (calls.incrementAndGet() <= 2) {
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
                while (release.getCount() > 0 && System.nanoTime() < deadline) {
                    try {
                        release.await(10, TimeUnit.MILLISECONDS);
                    } catch (InterruptedException timeout) {
                        // Model a discovery implementation that ignores interruption.
                    }
                }
            }
            return List.of(WorkerHost.of("recovered", 80));
        });
        create(WebClient.create());
        try {
            pool.refresh();
            pool.refresh();
            pool.refresh();
            assertEquals(2, calls.get());
            assertEquals(0, pool.currentSize());
            release.countDown();
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (pool.currentSize() == 0 && System.nanoTime() < deadline) {
                pool.refresh();
                Thread.sleep(5);
            }
            assertEquals("http://recovered:80", pool.next());
        } finally {
            release.countDown();
        }
    }
}
