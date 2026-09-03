package org.flexlb.service.config.source;

import com.sun.net.httpserver.HttpServer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.config.parser.StandardConfigDocumentParser;
import org.flexlb.service.config.parser.V0ConfigDocumentParser;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;
import org.springframework.test.util.ReflectionTestUtils;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.flexlb.constant.DeploymentIdentityConstants.HIPPO_ROLE;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_APPLICATION_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_DEPLOYMENT_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_WORKSPACE_ID;
import static org.flexlb.constant.NacosConfigConstants.NACOS_SERVER_ADDR;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;

class UniConfigConfigSourceTest {

    private static final String CONFIG_PATH = "/v2/configs/modelstudio.spectrum.deployment."
            + "df4a7748.flexlb-test-wlcb.runtime.meta";
    private static final String INITIAL_CONFIG = """
            {"schemaVersion":1,"router":{"availabilityHysteresisPercent":9}}
            """;
    private static final String UPDATED_CONFIG = """
            {"schemaVersion":1,"router":{"availabilityHysteresisPercent":10}}
            """;

    private final ScheduledExecutorService executor = mock(ScheduledExecutorService.class);
    private final AtomicReference<Response> response = new AtomicReference<>(new Response(200, INITIAL_CONFIG));
    private final AtomicInteger requests = new AtomicInteger();
    private final AtomicReference<String> requestPath = new AtomicReference<>();
    private final AtomicReference<String> requestMethod = new AtomicReference<>();
    private HttpServer server;
    private UniConfigConfigSource source;
    private ConfigService configService;
    private Runnable poll;

    @BeforeEach
    void startAgent() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            requests.incrementAndGet();
            requestPath.set(exchange.getRequestURI().getRawPath());
            requestMethod.set(exchange.getRequestMethod());
            Response current = response.get();
            byte[] body = current.body().getBytes(StandardCharsets.UTF_8);
            try (exchange) {
                exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
                exchange.sendResponseHeaders(current.status(), body.length);
                exchange.getResponseBody().write(body);
            }
        });
        server.start();
    }

    @AfterEach
    void close() {
        if (configService != null) {
            configService.close();
        }
        if (source != null) {
            source.close();
        }
        server.stop(0);
    }

    @Test
    void fetchesRawDocumentsAndHotUpdatesThroughBothExistingParsers() throws Exception {
        String legacyConfig = "{\"schemaVersion\":0,\"enableQueueing\":true}";
        response.set(new Response(200, legacyConfig));
        initializeSource();
        initializeConfigService();

        assertThat(source.name()).isEqualTo("UniConfig");
        assertThat(source.load()).isEqualTo(legacyConfig);
        assertThat(requestPath.get()).isEqualTo(CONFIG_PATH);
        assertThat(requestMethod.get()).isEqualTo("GET");
        assertThat(configService.loadBalanceConfig().isQueue()).isTrue();
        AtomicInteger updates = new AtomicInteger();
        configService.addUpdateListener(config -> updates.incrementAndGet());
        FlexlbConfig original = configService.loadBalanceConfig();

        poll.run();

        assertThat(configService.loadBalanceConfig()).isSameAs(original);
        assertThat(updates).hasValue(1);
        String currentConfig = """
                {"schemaVersion":1,"scheduler":{"type":"DIRECT"},
                "dispatcher":{"type":"NON_BATCH"},"router":{"availabilityHysteresisPercent":9}}
                """;
        response.set(new Response(200, currentConfig));
        poll.run();

        assertThat(source.load()).isEqualTo(currentConfig);
        assertThat(configService.loadBalanceConfig().isDirect()).isTrue();
        assertThat(configService.loadBalanceConfig().getRouter().getAvailabilityHysteresisPercent()).isEqualTo(9);
        assertThat(updates).hasValue(2);
        poll.run();
        assertThat(updates).hasValue(2);

        configService.close();
        int requestsBeforeClose = requests.get();
        poll.run();
        assertThat(requests).hasValue(requestsBeforeClose);
        verify(executor).shutdownNow();
    }

    @ParameterizedTest
    @ValueSource(strings = {"not-json", "{\"schemaVersion\":1,\"enableFallback\":true}"})
    void uniConfigExcludesNacosAndEnvironmentBehaviorButPreservesModelTopology(String environmentConfig)
            throws Exception {
        com.alibaba.nacos.api.config.ConfigService nacosClient =
                mock(com.alibaba.nacos.api.config.ConfigService.class);
        spectrumEnvironment()
                .set("FLEXLB_UNICONF_ENABLE", " TrUe ")
                .set(NACOS_SERVER_ADDR, "unreachable-nacos:8848")
                .set("FLEXLB_CONFIG", environmentConfig)
                .set("MODEL_SERVICE_CONFIG", "{\"service_id\":\"test-model\",\"role_endpoints\":[]}")
                .execute(() -> {
                    DeploymentIdentity identity = new DeploymentIdentity();
                    EnvironmentConfigSource environmentSource = new EnvironmentConfigSource();
                    environmentSource.initialize();
                    NacosConfigSource nacosSource = new NacosConfigSource(identity);
                    ReflectionTestUtils.setField(nacosSource, "client", nacosClient);
                    nacosSource.initialize();
                    prepareSource(new UniConfigConfigSource(identity));
                    source.initialize();
                    initializeConfigService();
                });

        assertThat(configService.loadBalanceConfig().getRouter().getAvailabilityHysteresisPercent()).isEqualTo(9);
        assertThat(configService.loadBalanceConfig().isEnableFallback()).isFalse();
        assertThat(configService.modelServiceConfig().getServiceId()).isEqualTo("test-model");
        verifyNoInteractions(nacosClient);
    }

    @Test
    void disabledSourceDoesNotRequireSpectrumOrContactAgent() throws Exception {
        source = new EnvironmentVariables("FLEXLB_UNICONF_ENABLE", "false", HIPPO_ROLE, "legacy-flexlb")
                .set("UNICONF_ENABLE", "true")
                .remove(SPECTRUM_WORKSPACE_ID)
                .remove(SPECTRUM_APPLICATION_NAME)
                .remove(SPECTRUM_DEPLOYMENT_NAME)
                .remove(NACOS_SERVER_ADDR)
                .execute(() -> new UniConfigConfigSource(new DeploymentIdentity()));

        source.initialize();

        assertThat(source.load()).isNull();
        assertThat(source).extracting("pollExecutor").isNull();
        assertThat(requests).hasValue(0);
    }

    @Test
    void enabledSourceRequiresSpectrumIdentity() {
        EnvironmentVariables environment = new EnvironmentVariables("FLEXLB_UNICONF_ENABLE", "true", HIPPO_ROLE, "legacy-flexlb")
                .remove(SPECTRUM_WORKSPACE_ID)
                .remove(SPECTRUM_APPLICATION_NAME)
                .remove(SPECTRUM_DEPLOYMENT_NAME);

        assertThatThrownBy(() -> environment.execute(() -> new UniConfigConfigSource(new DeploymentIdentity())))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("UniConfig requires a Spectrum deployment identity");
    }

    @ParameterizedTest
    @ValueSource(ints = {404, 503})
    void failsStartupWhenSelectedConfigCannotBeFetched(int status) {
        response.set(new Response(status, "{\"error\":\"config unavailable\"}"));

        assertThatThrownBy(this::initializeSource)
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("Failed to initialize UniConfig")
                .hasRootCauseInstanceOf(IOException.class)
                .hasRootCauseMessage("UniConfig returned HTTP " + status + " for http://127.0.0.1:"
                        + server.getAddress().getPort() + CONFIG_PATH);

        verify(executor).shutdownNow();
    }

    @Test
    void failsStartupAndStopsPollingWhenInitialDocumentIsInvalid() throws Exception {
        response.set(new Response(200, "{\"schemaVersion\":1,\"unknown\":true}"));
        initializeSource();

        assertThatThrownBy(this::initializeConfigService)
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("UniConfig");

        verify(executor).shutdownNow();
    }

    @ParameterizedTest
    @ValueSource(ints = {404, 503})
    void retainsLastGoodConfigOnHttpErrorsAndRecovers(int status) throws Exception {
        initializeSource();
        initializeConfigService();
        FlexlbConfig original = configService.loadBalanceConfig();
        response.set(new Response(status, "{}"));

        poll.run();

        assertThat(source.load()).isEqualTo(INITIAL_CONFIG);
        assertThat(configService.loadBalanceConfig()).isSameAs(original);
        response.set(new Response(200, UPDATED_CONFIG));
        poll.run();
        assertThat(configService.loadBalanceConfig().getRouter().getAvailabilityHysteresisPercent()).isEqualTo(10);
    }

    @ParameterizedTest
    @ValueSource(strings = {"{", "{\"schemaVersion\":1,\"unknown\":true}",
            "{\"schemaVersion\":1,\"scheduler\":{\"type\":\"DIRECT\"},\"dispatcher\":{\"type\":\"BATCH\"}}"})
    void retainsLastGoodConfigOnInvalidUpdatesAndRecovers(String invalidConfig) throws Exception {
        initializeSource();
        initializeConfigService();
        FlexlbConfig original = configService.loadBalanceConfig();
        response.set(new Response(200, invalidConfig));

        poll.run();

        assertThat(configService.loadBalanceConfig()).isSameAs(original);
        response.set(new Response(200, UPDATED_CONFIG));
        poll.run();
        assertThat(configService.loadBalanceConfig().getRouter().getAvailabilityHysteresisPercent()).isEqualTo(10);
    }

    private EnvironmentVariables spectrumEnvironment() {
        return new EnvironmentVariables(
                "FLEXLB_UNICONF_ENABLE", "true",
                SPECTRUM_WORKSPACE_ID, "df4a7748",
                SPECTRUM_APPLICATION_NAME, "flexlb-test",
                SPECTRUM_DEPLOYMENT_NAME, "flexlb-test-wlcb")
                .set(NACOS_SERVER_ADDR, null)
                .set("FLEXLB_CONFIG", null)
                .set("MODEL_SERVICE_CONFIG", null);
    }

    private void prepareSource(UniConfigConfigSource newSource) {
        source = newSource;
        assertThat(source).extracting("configUri").isEqualTo(URI.create("http://127.0.0.1:18080" + CONFIG_PATH));
        ReflectionTestUtils.setField(source, "configUri",
                URI.create("http://127.0.0.1:" + server.getAddress().getPort() + CONFIG_PATH));
        ReflectionTestUtils.setField(source, "pollExecutor", executor);
    }

    private void initializeSource() throws Exception {
        spectrumEnvironment().execute(() -> {
            prepareSource(new UniConfigConfigSource(new DeploymentIdentity()));
            source.initialize();
        });
    }

    private void initializeConfigService() {
        configService = new ConfigService(List.of(new StandardConfigDocumentParser(), new V0ConfigDocumentParser()));
        ArgumentCaptor<Runnable> pollCaptor = ArgumentCaptor.forClass(Runnable.class);
        verify(executor).scheduleWithFixedDelay(pollCaptor.capture(), eq(60L), eq(60L), eq(TimeUnit.SECONDS));
        poll = pollCaptor.getValue();
    }

    private record Response(int status, String body) {}
}
