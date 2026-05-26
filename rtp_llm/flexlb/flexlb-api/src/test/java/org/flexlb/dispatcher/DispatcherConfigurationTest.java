package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceHostListener;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.fail;

class DispatcherConfigurationTest {

    private static final List<BatchEndpointSpec> SPECS = new BatchEndpointRegistry().batchSpecs();

    @Test
    void refresherBeanCreatedWhenEnabled() {
        DispatchConfig cfg = DispatchConfig.fromJson(
                "{\"enabled\":true,\"fePoolServiceId\":\"com.rtp_llm.fe\"}");
        DispatcherConfiguration conf = new DispatcherConfiguration();
        StubServiceDiscovery discovery = new StubServiceDiscovery("com.rtp_llm.fe");

        DispatcherFePoolRefresher refresher = conf.dispatcherFePoolRefresher(cfg, discovery);

        assertNotNull(refresher, "enabled dispatcher must register a refresher bean");
        assertEquals(1, discovery.getHostsCalls,
                "refresher constructor must seed the pool by calling getHosts once");
        assertNotNull(discovery.registeredListener,
                "refresher must subscribe via listen() for push-based fast-path updates");
    }

    @Test
    void noRefresherBeanWhenDisabled() {
        DispatchConfig cfg = DispatchConfig.fromJson(null);
        DispatcherConfiguration conf = new DispatcherConfiguration();

        DispatcherFePoolRefresher refresher = conf.dispatcherFePoolRefresher(cfg, new FailingServiceDiscovery());

        assertNull(refresher, "disabled dispatcher must not register a refresher bean");
    }

    @Test
    void buildsRouterWhenEnabled() {
        DispatchConfig cfg = DispatchConfig.fromJson(
                "{\"enabled\":true,\"fePoolServiceId\":\"com.rtp_llm.fe\"}");
        DispatcherConfiguration conf = new DispatcherConfiguration();
        DispatcherFePoolRefresher refresher = new DispatcherFePoolRefresher(
                new StubServiceDiscovery("com.rtp_llm.fe"), "com.rtp_llm.fe");

        RouterFunction<ServerResponse> routes = conf.dispatcherRoutes(
                cfg, new ObjectMapper(), WebClient.builder(), provider(refresher), SPECS);

        assertNotNull(routes);
    }

    @Test
    void noRouterWhenDisabled() {
        DispatchConfig cfg = DispatchConfig.fromJson(null);
        DispatcherConfiguration conf = new DispatcherConfiguration();

        assertNull(conf.dispatcherRoutes(
                cfg, new ObjectMapper(), WebClient.builder(), absentProvider(), SPECS));
    }

    @Test
    void pollFillsPoolAfterColdSeed() {
        // Boot path: service discovery cache is cold, so the constructor's getHosts() returns
        // []. The refresher must lift the pool to a healthy snapshot on the next poll once
        // discovery has populated. This is the exact production timeline that produced
        // seedHosts=0 + IllegalStateException("no FE endpoints available") before the fix.
        StubServiceDiscovery discovery = new StubServiceDiscovery("com.rtp_llm.fe");
        DispatcherFePoolRefresher refresher = new DispatcherFePoolRefresher(discovery, "com.rtp_llm.fe");
        assertEquals(0, refresher.currentSize(),
                "cold seed: refresher must accept an empty initial pool without throwing");

        discovery.setHosts(List.of(WorkerHost.of("10.0.0.1", 8088), WorkerHost.of("10.0.0.2", 8088)));
        refresher.refresh();

        assertEquals(2, refresher.currentSize(),
                "next poll must surface the freshly populated discovery snapshot");
    }

    @Test
    void pollPathRedirectsTrafficToFreshHosts() throws Exception {
        runRedirectScenario((discovery, refresher, freshHost) -> {
            // Service discovery sees the new host on the next periodic poll.
            discovery.setHosts(List.of(freshHost));
            refresher.refresh();
        });
    }

    @Test
    void listenerPathRedirectsTrafficToFreshHosts() throws Exception {
        runRedirectScenario((discovery, refresher, freshHost) -> {
            // Service discovery pushes the change via the registered listener — no poll
            // required. Mirrors how master's EngineAddressNameResolver consumes push events.
            discovery.pushHosts(List.of(freshHost));
        });
    }

    /**
     * Shared scenario for both refresh paths: start with one FE host, swap to a fresh host
     * via the supplied trigger, then assert traffic lands on the fresh host. Lets the two
     * tests above differ only in <em>how</em> the swap is signalled (poll vs listener push).
     */
    private void runRedirectScenario(RedirectTrigger trigger) throws Exception {
        MockWebServer staleFe = new MockWebServer();
        MockWebServer freshFe = new MockWebServer();
        staleFe.start();
        freshFe.start();
        try {
            freshFe.enqueue(new MockResponse()
                    .setHeader("Content-Type", "application/json")
                    .setBody("{\"response_batch\":[{\"response\":\"ok\"}]}"));

            DispatchConfig cfg = DispatchConfig.fromJson(
                    "{\"enabled\":true,\"subBatch\":\"size:1\","
                            + "\"fePoolServiceId\":\"com.rtp_llm.fe\"}");
            StubServiceDiscovery discovery = new StubServiceDiscovery("com.rtp_llm.fe",
                    WorkerHost.of(staleFe.getHostName(), staleFe.getPort()));
            DispatcherFePoolRefresher refresher = new DispatcherFePoolRefresher(discovery, "com.rtp_llm.fe");

            DispatcherConfiguration conf = new DispatcherConfiguration();
            RouterFunction<ServerResponse> routes = conf.dispatcherRoutes(
                    cfg, new ObjectMapper(), WebClient.builder(), provider(refresher), SPECS);

            trigger.run(discovery, refresher,
                    WorkerHost.of(freshFe.getHostName(), freshFe.getPort()));

            WebTestClient client = WebTestClient.bindToRouterFunction(routes).build();
            client.post().uri("/dispatcher/batch_infer")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue("{\"prompt_batch\":[\"x\"]}")
                    .exchange()
                    .expectStatus().isOk();

            assertEquals(0, countBatchInferRequests(staleFe),
                    "stale host must not receive batch traffic after pool swap (health probes ok)");
            assertEquals(1, countBatchInferRequests(freshFe),
                    "fresh host must receive the post-swap batch_infer request");
        } finally {
            staleFe.shutdown();
            freshFe.shutdown();
        }
    }

    @FunctionalInterface
    private interface RedirectTrigger {
        void run(StubServiceDiscovery discovery, DispatcherFePoolRefresher refresher, WorkerHost freshHost);
    }

    /**
     * Drains a MockWebServer's request queue and counts only {@code /batch_infer} requests, so
     * background {@code /frontend_health} probes from {@link FeHealthChecker} are not mistaken for
     * batch traffic in tests asserting which FE got hit.
     */
    private static int countBatchInferRequests(MockWebServer server) throws InterruptedException {
        int count = 0;
        RecordedRequest req;
        while ((req = server.takeRequest(50, TimeUnit.MILLISECONDS)) != null) {
            if (req.getPath() != null && req.getPath().startsWith("/batch_infer")) {
                count++;
            }
        }
        return count;
    }

    private static <T> ObjectProvider<T> provider(T instance) {
        return new SingletonObjectProvider<>(instance);
    }

    private static <T> ObjectProvider<T> absentProvider() {
        return new SingletonObjectProvider<>(null);
    }

    /** Minimal ObjectProvider stand-in for tests — Spring's own ObjectProvider has no public ctor. */
    private static final class SingletonObjectProvider<T> implements ObjectProvider<T> {
        private final T instance;

        SingletonObjectProvider(T instance) {
            this.instance = instance;
        }

        @Override
        public T getObject() {
            if (instance == null) {
                throw new IllegalStateException("no bean");
            }
            return instance;
        }

        @Override
        public T getObject(Object... args) {
            return getObject();
        }

        @Override
        public T getIfAvailable() {
            return instance;
        }

        @Override
        public T getIfUnique() {
            return instance;
        }

        @Override
        public void ifAvailable(Consumer<T> dependencyConsumer) {
            if (instance != null) {
                dependencyConsumer.accept(instance);
            }
        }

        @Override
        public void ifUnique(Consumer<T> dependencyConsumer) {
            ifAvailable(dependencyConsumer);
        }
    }

    /** Stub that returns a controllable host list for one expected service id and lets
     * the test trigger a push via the registered listener. */
    private static final class StubServiceDiscovery implements ServiceDiscovery {
        final String expectedId;
        final AtomicReference<List<WorkerHost>> hosts;
        int getHostsCalls = 0;
        ServiceHostListener registeredListener = null;

        StubServiceDiscovery(String expectedId, WorkerHost... initial) {
            this.expectedId = expectedId;
            this.hosts = new AtomicReference<>(List.of(initial));
        }

        /** Update the snapshot returned by {@link #getHosts}. Used by poll-path tests. */
        void setHosts(List<WorkerHost> updated) {
            this.hosts.set(updated);
        }

        /** Update the snapshot AND deliver it to the registered listener. Used by push-path tests. */
        void pushHosts(List<WorkerHost> updated) {
            this.hosts.set(updated);
            if (registeredListener != null) {
                registeredListener.onHostsChanged(updated);
            }
        }

        @Override
        public List<WorkerHost> getHosts(String address) {
            assertEquals(expectedId, address);
            getHostsCalls++;
            return hosts.get();
        }

        @Override
        public void listen(String address, ServiceHostListener listener) {
            assertEquals(expectedId, address);
            this.registeredListener = listener;
        }

        @Override
        public void shutdown() {}
    }

    /** Strict stub that fails the test if discovery is touched at all. */
    private static final class FailingServiceDiscovery implements ServiceDiscovery {
        @Override
        public List<WorkerHost> getHosts(String address) {
            return fail("disabled dispatcher must not query ServiceDiscovery (getHosts " + address + ")");
        }

        @Override
        public void listen(String address, ServiceHostListener listener) {
            fail("disabled dispatcher must not subscribe to ServiceDiscovery (listen " + address + ")");
        }

        @Override
        public void shutdown() {
            fail("disabled dispatcher must not shut down ServiceDiscovery");
        }
    }
}
