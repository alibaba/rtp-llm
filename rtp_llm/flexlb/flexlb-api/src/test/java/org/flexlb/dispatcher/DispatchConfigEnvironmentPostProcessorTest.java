package org.flexlb.dispatcher;

import org.flexlb.service.grace.ActiveRequestCounter;
import org.junit.jupiter.api.Test;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.WebApplicationType;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Import;
import org.springframework.boot.logging.DeferredLogs;
import org.springframework.mock.env.MockEnvironment;
import org.springframework.mock.http.server.reactive.MockServerHttpRequest;
import org.springframework.mock.web.server.MockServerWebExchange;
import org.springframework.web.reactive.function.server.HandlerStrategies;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;

import java.util.List;

import static org.flexlb.dispatcher.DispatchConfigEnvironmentPostProcessor.ENABLE_PROPERTY;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.mock;

class DispatchConfigEnvironmentPostProcessorTest {

    private final DispatchConfigEnvironmentPostProcessor epp =
            new DispatchConfigEnvironmentPostProcessor(new DeferredLogs());

    @Test
    void jsonOnlyConfigEnablesDispatcher() {
        MockEnvironment env = new MockEnvironment();
        env.setProperty("DISPATCH_CONFIG", "{\"fePoolServiceId\":\"rtp_llm.frontend.service\"}");

        epp.postProcessEnvironment(env, null);

        assertEquals("rtp_llm.frontend.service", env.getProperty(ENABLE_PROPERTY),
                "DISPATCH_CONFIG.fePoolServiceId must populate the enable property on its own");
    }

    @Test
    void explicitEnablePropertyWins() {
        MockEnvironment env = new MockEnvironment();
        env.setProperty(ENABLE_PROPERTY, "explicit.service");
        env.setProperty("DISPATCH_CONFIG", "{\"fePoolServiceId\":\"json.service\"}");

        epp.postProcessEnvironment(env, null);

        assertEquals("explicit.service", env.getProperty(ENABLE_PROPERTY),
                "an explicit DISPATCH_FE_POOL_SERVICE_ID must not be overridden by the JSON");
    }

    @Test
    void malformedConfigDoesNotEnableAndDoesNotThrow() {
        MockEnvironment env = new MockEnvironment();
        env.setProperty("DISPATCH_CONFIG", "{not valid json");

        epp.postProcessEnvironment(env, null);

        assertNull(env.getProperty(ENABLE_PROPERTY),
                "a malformed DISPATCH_CONFIG must leave the dispatcher disabled, not crash boot");
    }

    @Test
    void blankFePoolServiceIdDoesNotEnable() {
        MockEnvironment env = new MockEnvironment();
        env.setProperty("DISPATCH_CONFIG", "{\"fePoolServiceId\":\"  \"}");

        epp.postProcessEnvironment(env, null);

        assertNull(env.getProperty(ENABLE_PROPERTY));
    }

    @Test
    void springFactoriesJsonOnlyBootCreatesDispatcherBeanAndRoute() {
        SpringApplication application = new SpringApplication(JsonOnlyDispatcherTestApplication.class);
        application.setWebApplicationType(WebApplicationType.NONE);
        application.setRegisterShutdownHook(false);
        application.setLogStartupInfo(false);

        try (ConfigurableApplicationContext context = application.run(
                "--spring.main.banner-mode=off",
                "--spring.main.web-application-type=none",
                "--logging.level.root=OFF",
                "--DISPATCH_CONFIG={\"fePoolServiceId\":\"json-only.service\"}")) {
            assertEquals("json-only.service", context.getEnvironment().getProperty(ENABLE_PROPERTY));
            assertNotNull(context.getBean(DispatchRouter.class),
                    "spring.factories must run before the dispatcher condition is evaluated");

            @SuppressWarnings("unchecked")
            RouterFunction<ServerResponse> routes =
                    (RouterFunction<ServerResponse>) context.getBean("dispatcherRoutes");
            ServerRequest request = ServerRequest.create(
                    MockServerWebExchange.from(MockServerHttpRequest.post("/dispatcher/batch_infer").build()),
                    HandlerStrategies.withDefaults().messageReaders());
            assertNotNull(routes.route(request).block(),
                    "JSON-only activation must publish the real dispatcher route table");
        }
    }

    @SpringBootConfiguration(proxyBeanMethods = false)
    @Import(DispatchRouter.class)
    static class JsonOnlyDispatcherTestApplication {

        @Bean
        BatchHandler batchHandler() {
            return mock(BatchHandler.class);
        }

        @Bean
        PassthroughClient passthroughClient() {
            return mock(PassthroughClient.class);
        }

        @Bean
        DispatcherInspectionHandler dispatcherInspectionHandler() {
            return mock(DispatcherInspectionHandler.class);
        }

        @Bean
        ActiveRequestCounter activeRequestCounter() {
            return mock(ActiveRequestCounter.class);
        }

        @Bean
        List<BatchEndpointSpec> batchEndpointSpecs() {
            return BatchEndpointSpec.SPECS;
        }

        @Bean
        RouterFunction<ServerResponse> dispatcherRoutes(DispatchRouter router) {
            return router.routes();
        }
    }
}
