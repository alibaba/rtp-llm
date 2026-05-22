package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

class DispatcherConfigurationTest {

    @Test
    void buildsRouterWhenEnabled() {
        DispatchConfig cfg = DispatchConfig.fromJson(
                "{\"enabled\":true,\"subBatchSize\":5,"
                        + "\"feRequestTimeoutMs\":3000,\"fePoolAddresses\":[\"http://a:8088\"]}");
        DispatcherConfiguration conf = new DispatcherConfiguration();
        RouterFunction<ServerResponse> routes =
                conf.dispatcherRoutes(cfg, new ObjectMapper(), WebClient.builder());
        assertNotNull(routes);
    }

    @Test
    void noRouterWhenDisabled() {
        DispatchConfig cfg = DispatchConfig.fromJson(null);
        DispatcherConfiguration conf = new DispatcherConfiguration();
        assertNull(conf.dispatcherRoutes(cfg, new ObjectMapper(), WebClient.builder()));
    }
}
