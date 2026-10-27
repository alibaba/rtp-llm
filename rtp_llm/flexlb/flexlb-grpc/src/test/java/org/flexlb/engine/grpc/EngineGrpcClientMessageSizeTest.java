package org.flexlb.engine.grpc;

import io.grpc.Server;
import io.grpc.Status;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.beans.BeansException;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.core.SpringProperties;
import org.springframework.core.env.AbstractEnvironment;
import org.springframework.core.env.MapPropertySource;
import org.springframework.core.env.SystemEnvironmentPropertySource;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

class EngineGrpcClientMessageSizeTest {
    private static final int MIB = 1024 * 1024;
    private static final String LOOPBACK = "127.0.0.1";
    private static final EngineRpcService.CacheStatusPB LARGE_CACHE_RESPONSE = cacheResponse(650_000);
    private final ThreadPoolExecutor executor = new ThreadPoolExecutor(
            1, 1, 1, TimeUnit.MINUTES, new LinkedBlockingQueue<>());
    private final NioEventLoopGroup eventLoopGroup = new NioEventLoopGroup(1);
    private final AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext();
    private Server server;
    private EngineGrpcClient client;
    private EngineRpcService.CacheStatusPB cacheResponse = LARGE_CACHE_RESPONSE;

    @BeforeEach
    void startServer() throws Exception {
        assertTrue(LARGE_CACHE_RESPONSE.getSerializedSize() > 8 * MIB);
        assertTrue(LARGE_CACHE_RESPONSE.getSerializedSize() < 16 * MIB);
        server = NettyServerBuilder.forAddress(new java.net.InetSocketAddress(LOOPBACK, 0))
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void getCacheStatus(EngineRpcService.CacheVersionPB request,
                                               StreamObserver<EngineRpcService.CacheStatusPB> observer) {
                        reply(observer, cacheResponse);
                    }

                    @Override
                    public void getWorkerStatus(EngineRpcService.StatusVersionPB request,
                                                StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
                        reply(observer, EngineRpcService.WorkerStatusPB.newBuilder()
                                .setRole("x".repeat(9 * MIB)).build());
                    }
                })
                .addService(new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
                    @Override
                    public void getCacheStatus(EngineRpcService.CacheVersionPB request,
                                               StreamObserver<EngineRpcService.CacheStatusPB> observer) {
                        reply(observer, LARGE_CACHE_RESPONSE);
                    }
                }).build().start();
    }

    @AfterEach
    void close() throws InterruptedException {
        context.close();
        if (server != null) {
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
        executor.shutdownNow();
        executor.awaitTermination(5, TimeUnit.SECONDS);
        eventLoopGroup.shutdownGracefully(0, 1, TimeUnit.SECONDS).await(5, TimeUnit.SECONDS);
    }

    @Test
    void defaultCacheLimitStillRejectsSnapshotsLargerThanEightMib() throws Exception {
        startClient(Map.of(), false);

        assertResourceExhausted(cacheStatus());
    }

    @Test
    void configuredSmallerLimitRejectsSnapshotsBelowTheDefaultLimit() throws Exception {
        cacheResponse = cacheResponse(100);
        assertTrue(cacheResponse.getSerializedSize() > 1024);
        assertTrue(cacheResponse.getSerializedSize() < 8 * MIB);
        startClient(Map.of(EngineGrpcClient.CACHE_MAX_INBOUND_MESSAGE_BYTES_PROPERTY, 1024), false);

        assertResourceExhausted(cacheStatus());
    }

    @Test
    void configuredCacheLimitAcceptsLargeSnapshotsWithoutChangingOtherLimits() throws Exception {
        startClient(Map.of(EngineGrpcClient.CACHE_MAX_INBOUND_MESSAGE_BYTES_PROPERTY, 64 * MIB), false);

        for (int i = 0; i < 2; i++) {
            assertEquals(LARGE_CACHE_RESPONSE, cacheStatus().get(10, TimeUnit.SECONDS));
        }
        assertResourceExhausted(client.getWorkerStatusAsync(LOOPBACK, server.getPort(),
                EngineRpcService.StatusVersionPB.getDefaultInstance(), 10000));
        assertEquals(LARGE_CACHE_RESPONSE, client.getMultimodalCacheStatusAsync(
                LOOPBACK, server.getPort(), request(), 10000).get(10, TimeUnit.SECONDS));
    }

    @Test
    void environmentOverrideAppliesToChannelsCreatedDuringInitialDiscovery() throws Exception {
        startClient(Map.of("FLEXLB_ENGINE_GRPC_CACHE_MAX_INBOUND_MESSAGE_BYTES", String.valueOf(64 * MIB)), true);

        assertEquals(LARGE_CACHE_RESPONSE, cacheStatus().get(10, TimeUnit.SECONDS));
    }

    @Test
    void processEnvironmentOverrideWorksWhenSpringEnvironmentLookupIsDisabled(@TempDir Path directory) throws Exception {
        Path output = directory.resolve("environment-process.log");
        var command = new ProcessBuilder(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "-cp", System.getProperty("surefire.test.class.path", System.getProperty("java.class.path")),
                EnvironmentProcessProbe.class.getName());
        command.environment().put("FLEXLB_ENGINE_GRPC_CACHE_MAX_INBOUND_MESSAGE_BYTES", String.valueOf(64 * MIB));
        command.redirectErrorStream(true).redirectOutput(output.toFile());
        Process process = command.start();
        try {
            assertTrue(process.waitFor(30, TimeUnit.SECONDS), "environment probe timed out");
            assertEquals(0, process.exitValue(), Files.readString(output));
        } finally {
            if (process.isAlive()) {
                process.destroyForcibly().waitFor(5, TimeUnit.SECONDS);
            }
        }
    }

    @Test
    void vitCacheLimitRemainsIndependent() throws Exception {
        startClient(Map.of(
                EngineGrpcClient.CACHE_MAX_INBOUND_MESSAGE_BYTES_PROPERTY, 64 * MIB,
                EngineGrpcClient.VIT_CACHE_MAX_INBOUND_MESSAGE_BYTES_PROPERTY, MIB), false);

        assertEquals(LARGE_CACHE_RESPONSE, cacheStatus().get(10, TimeUnit.SECONDS));
        assertResourceExhausted(client.getMultimodalCacheStatusAsync(
                LOOPBACK, server.getPort(), request(), 10000));
    }

    @ParameterizedTest
    @ValueSource(strings = {"0", "-1", "2147483648", "64MiB"})
    void invalidCacheLimitsFailStartup(String value) {
        BeansException failure = assertThrows(BeansException.class,
                () -> startClient(Map.of(EngineGrpcClient.CACHE_MAX_INBOUND_MESSAGE_BYTES_PROPERTY, value), false));

        assertTrue(failure.getMostSpecificCause().getMessage().contains(
                value.equals("0") || value.equals("-1") ? "must be positive" : value));
    }

    private void startClient(Map<String, Object> properties, boolean initialDiscovery) {
        context.getEnvironment().getPropertySources().addFirst(initialDiscovery
                ? new SystemEnvironmentPropertySource("test-environment", properties)
                : new MapPropertySource("test-properties", properties));
        context.registerBean("managedChannelThreadPoolExecutor", ThreadPoolExecutor.class, () -> executor);
        context.registerBean("managedChannelEventLoopGroup", EventLoopGroup.class, () -> eventLoopGroup);
        context.registerBean(CustomNameResolver.class, () -> listener -> {
            if (initialDiscovery) {
                listener.onAddressUpdate(List.of(LOOPBACK + ":" + (server.getPort() - 1)));
            }
        });
        context.registerBean(GrpcReporter.class, () -> mock(GrpcReporter.class));
        context.registerBean(EngineGrpcClient.class);
        context.refresh();
        client = context.getBean(EngineGrpcClient.class);
    }

    private CompletableFuture<EngineRpcService.CacheStatusPB> cacheStatus() {
        return client.getCacheStatusAsync(LOOPBACK, server.getPort(), request(), 10000);
    }

    private static EngineRpcService.CacheVersionPB request() {
        return EngineRpcService.CacheVersionPB.newBuilder().setLatestCacheVersion(-1).setNeedCacheKeys(true).build();
    }

    private static void assertResourceExhausted(CompletableFuture<?> future) {
        ExecutionException failure = assertThrows(ExecutionException.class, () -> future.get(10, TimeUnit.SECONDS));
        assertEquals(Status.Code.RESOURCE_EXHAUSTED, Status.fromThrowable(failure.getCause()).getCode());
    }

    private static <T> void reply(StreamObserver<T> observer, T value) {
        observer.onNext(value);
        observer.onCompleted();
    }

    private static EngineRpcService.CacheStatusPB cacheResponse(int keys) {
        var response = EngineRpcService.CacheStatusPB.newBuilder().setVersion(1).setBlockSize(512);
        for (long i = 0; i < keys; i++) {
            response.putCacheKeys(Long.MIN_VALUE + i, true);
        }
        return response.build();
    }

    public static final class EnvironmentProcessProbe {
        public static void main(String[] args) throws Exception {
            SpringProperties.setFlag(AbstractEnvironment.IGNORE_GETENV_PROPERTY_NAME);
            var test = new EngineGrpcClientMessageSizeTest();
            try {
                assertTrue(test.context.getEnvironment().getSystemEnvironment().isEmpty());
                test.startServer();
                test.startClient(Map.of(), false);
                assertEquals(LARGE_CACHE_RESPONSE, test.cacheStatus().get(10, TimeUnit.SECONDS));
            } finally {
                test.close();
            }
        }
    }
}
