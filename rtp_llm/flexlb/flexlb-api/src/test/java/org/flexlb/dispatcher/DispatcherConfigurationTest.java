package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceHostListener;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.fail;

class DispatcherConfigurationTest {

    @Test
    void buildsRouterWhenEnabled() {
        DispatchConfig cfg = DispatchConfig.fromJson(
                "{\"enabled\":true,\"subBatchSize\":5,"
                        + "\"feRequestTimeoutMs\":3000,\"fePoolServiceId\":\"com.rtp_llm.fe\"}");
        DispatcherConfiguration conf = new DispatcherConfiguration();
        RouterFunction<ServerResponse> routes = conf.dispatcherRoutes(
                cfg, new ObjectMapper(), WebClient.builder(), new StubServiceDiscovery("com.rtp_llm.fe"));
        assertNotNull(routes);
    }

    @Test
    void noRouterWhenDisabled() {
        DispatchConfig cfg = DispatchConfig.fromJson(null);
        DispatcherConfiguration conf = new DispatcherConfiguration();
        assertNull(conf.dispatcherRoutes(
                cfg, new ObjectMapper(), WebClient.builder(), new FailingServiceDiscovery()));
    }

    @Test
    void subscribesAndSeedsFromDiscovery() {
        DispatchConfig cfg = DispatchConfig.fromJson(
                "{\"enabled\":true,\"fePoolServiceId\":\"com.rtp_llm.fe\"}");
        StubServiceDiscovery discovery = new StubServiceDiscovery("com.rtp_llm.fe",
                WorkerHost.of("10.0.0.1", 8088));
        DispatcherConfiguration conf = new DispatcherConfiguration();
        conf.dispatcherRoutes(cfg, new ObjectMapper(), WebClient.builder(), discovery);

        assertEquals(1, discovery.getHostsCalls);
        assertNotNull(discovery.registeredListener, "dispatcher must subscribe to host changes");
    }

    @Test
    void listenerCallbackRedirectsTrafficToFreshHosts() throws Exception {
        MockWebServer staleFe = new MockWebServer();
        MockWebServer freshFe = new MockWebServer();
        staleFe.start();
        freshFe.start();
        try {
            freshFe.enqueue(new MockResponse()
                    .setHeader("Content-Type", "application/json")
                    .setBody("{\"response_batch\":[{\"response\":\"ok\"}]}"));

            DispatchConfig cfg = DispatchConfig.fromJson(
                    "{\"enabled\":true,\"subBatchSize\":1,\"fePoolServiceId\":\"com.rtp_llm.fe\"}");
            StubServiceDiscovery discovery = new StubServiceDiscovery("com.rtp_llm.fe",
                    WorkerHost.of(staleFe.getHostName(), staleFe.getPort()));
            DispatcherConfiguration conf = new DispatcherConfiguration();
            RouterFunction<ServerResponse> routes = conf.dispatcherRoutes(
                    cfg, new ObjectMapper(), WebClient.builder(), discovery);

            // Simulate a service-discovery push: the pool must immediately observe the new host
            // on subsequent next() calls; the request below must land on freshFe, not staleFe.
            discovery.registeredListener.onHostsChanged(
                    List.of(WorkerHost.of(freshFe.getHostName(), freshFe.getPort())));

            WebTestClient client = WebTestClient.bindToRouterFunction(routes).build();
            client.post().uri("/dispatcher/batch_infer")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue("{\"prompt_batch\":[\"x\"]}")
                    .exchange()
                    .expectStatus().isOk();

            assertEquals(0, staleFe.getRequestCount(), "stale host must not be hit after pool swap");
            assertEquals(1, freshFe.getRequestCount(), "fresh host must receive the post-swap request");
        } finally {
            staleFe.shutdown();
            freshFe.shutdown();
        }
    }

    /** Stub that returns a controllable host list for one expected service id. */
    private static final class StubServiceDiscovery implements ServiceDiscovery {
        final String expectedId;
        final AtomicReference<List<WorkerHost>> hosts;
        int getHostsCalls = 0;
        ServiceHostListener registeredListener = null;

        StubServiceDiscovery(String expectedId, WorkerHost... initial) {
            this.expectedId = expectedId;
            this.hosts = new AtomicReference<>(List.of(initial));
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
