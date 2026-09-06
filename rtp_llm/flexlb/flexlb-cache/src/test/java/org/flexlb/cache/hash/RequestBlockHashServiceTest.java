package org.flexlb.cache.hash;

import org.flexlb.cache.domain.BlockHashCalculationResult;
import org.flexlb.cache.domain.BlockHashConfig;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;

import java.util.ArrayList;
import java.util.List;

import static org.flexlb.cache.CacheMatchTestConfigurations.kvcm;
import static org.flexlb.cache.CacheMatchTestConfigurations.localSync;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class RequestBlockHashServiceTest {

    private final BlockHashConfigResolver configResolver = mock(BlockHashConfigResolver.class);
    private final BlockHashExecutor executor = mock(BlockHashExecutor.class);
    private final LocalStandbyHashService localStandbyHashService =
            mock(LocalStandbyHashService.class);
    private final RequestBlockHashService service =
            new RequestBlockHashService(
                    configResolver,
                    executor,
                    localStandbyHashService,
                    localSync(new ModelMetaConfig()));

    @Test
    void prefersProvidedBlockCacheKeys() {
        Request request = new Request();
        List<Long> keys = new ArrayList<>(List.of(11L, 22L));
        request.setBlockCacheKeys(keys);
        request.setBlockSize(2192);
        request.setInputIds(new int[]{1, 2});

        service.prepareBlockCacheKeys(contextFor(request)).block();

        assertSame(keys, request.getBlockCacheKeys());
        assertNull(request.getInputIds());
        verifyNoInteractions(configResolver, executor, localStandbyHashService);
    }

    @Test
    void rejectsProvidedKeysWithoutBlockSize() {
        Request request = new Request();
        request.setBlockCacheKeys(List.of(11L));

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> service.prepareBlockCacheKeys(contextFor(request)).block());

        assertEquals(
                "block_size must be greater than 0 when block_cache_keys are provided",
                error.getMessage());
    }

    @Test
    void calculatesKeysFromInputIds() {
        Request request = new Request();
        request.setInputIds(new int[]{1, 2, 3, 4});
        when(configResolver.resolve()).thenReturn(new BlockHashConfig(4L, 0));
        when(executor.calculate(request.getInputIds(), 4L, 0))
                .thenReturn(Mono.just(new BlockHashCalculationResult(
                        List.of(99L), 12, 34)));
        BalanceContext context = contextFor(request);

        service.prepareBlockCacheKeys(context).block();

        assertEquals(List.of(99L), request.getBlockCacheKeys());
        assertEquals(4L, request.getBlockSize());
        assertNull(request.getInputIds());
        assertEquals(12, context.getBlockHashQueueWaitTimeUs());
        assertEquals(34, context.getBlockHashExecutionTimeUs());
    }

    @Test
    void acceptsInputIdsWithoutACompleteBlock() {
        Request request = new Request();
        request.setInputIds(new int[]{1, 2});
        request.setBlockSize(4);
        BalanceContext context = contextFor(request);
        when(configResolver.resolve()).thenReturn(new BlockHashConfig(64L, 1));
        when(executor.calculate(request.getInputIds(), 4L, 1))
                .thenReturn(Mono.just(new BlockHashCalculationResult(
                        List.of(), 5, 8)));

        service.prepareBlockCacheKeys(context).block();

        assertEquals(List.of(), request.getBlockCacheKeys());
        assertNull(request.getInputIds());
    }

    @Test
    void failsWhenWorkerBlockHashConfigIsUnavailable() {
        Request request = new Request();
        request.setInputIds(new int[]{1});
        when(configResolver.resolve()).thenThrow(
                new IllegalStateException("block hash configuration is unavailable"));

        assertThrows(
                IllegalStateException.class,
                () -> service.prepareBlockCacheKeys(contextFor(request)).block());
    }

    @Test
    void calculatesLocalStandbyKeysWithoutWaitingOnPrimaryPath() {
        RequestBlockHashService standbyService = new RequestBlockHashService(
                configResolver,
                executor,
                localStandbyHashService,
                configurationWithLocalStandby(4096));
        Request request = new Request();
        int[] inputIds = new int[]{1, 2, 3, 4};
        request.setInputIds(inputIds);
        BalanceContext context = contextFor(request);
        when(configResolver.resolve()).thenReturn(new BlockHashConfig(2192, 1));
        when(executor.calculate(inputIds, 2192, 1))
                .thenReturn(Mono.just(new BlockHashCalculationResult(
                        List.of(11L, 22L), 12, 34)));

        standbyService.prepareBlockCacheKeys(context).block();

        assertEquals(List.of(11L, 22L), request.getBlockCacheKeys());
        assertEquals(2192, request.getBlockSize());
        assertNull(request.getLocalStandbyBlockCacheKeys());
        assertEquals(4096, request.getLocalStandbyBlockSize());
        assertNull(request.getInputIds());
        assertEquals(12, context.getBlockHashQueueWaitTimeUs());
        assertEquals(34, context.getBlockHashExecutionTimeUs());
        verify(localStandbyHashService).submit(request, inputIds, 4096, 1);
    }

    @Test
    void reusesPrimaryHashWhenLocalStandbyBlockSizeMatches() {
        RequestBlockHashService standbyService = new RequestBlockHashService(
                configResolver,
                executor,
                localStandbyHashService,
                configurationWithLocalStandby(2192));
        Request request = new Request();
        int[] inputIds = new int[]{1, 2, 3, 4};
        request.setInputIds(inputIds);
        List<Long> calculatedKeys = List.of(11L, 22L);
        List<Long> cacheableKeys = List.of(11L);
        when(configResolver.resolve()).thenReturn(new BlockHashConfig(2192, 1));
        when(executor.calculate(inputIds, 2192, 1))
                .thenReturn(Mono.just(new BlockHashCalculationResult(
                        calculatedKeys, 12, 34)));
        when(executor.cacheablePrefix(calculatedKeys, inputIds.length, 2192, 1))
                .thenReturn(cacheableKeys);

        standbyService.prepareBlockCacheKeys(contextFor(request)).block();

        assertSame(calculatedKeys, request.getBlockCacheKeys());
        assertSame(calculatedKeys, request.getLocalStandbyBlockCacheKeys());
        assertSame(cacheableKeys, request.getLocalStandbyCacheableBlockCacheKeys());
        assertEquals(2192, request.getLocalStandbyBlockSize());
        verify(localStandbyHashService, never()).submit(
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.any(),
                org.mockito.ArgumentMatchers.anyLong(),
                org.mockito.ArgumentMatchers.anyInt());
    }

    @Test
    void usesProvidedBlockCacheKeysForLocalStandby() {
        RequestBlockHashService standbyService = new RequestBlockHashService(
                configResolver,
                executor,
                localStandbyHashService,
                configurationWithLocalStandby(4096));
        Request request = new Request();
        List<Long> providedKeys = List.of(11L, 22L);
        request.setBlockCacheKeys(providedKeys);
        request.setBlockSize(2192);
        request.setInputIds(new int[]{1, 2, 3, 4});

        standbyService.prepareBlockCacheKeys(contextFor(request)).block();

        assertSame(providedKeys, request.getLocalStandbyBlockCacheKeys());
        assertSame(providedKeys, request.getLocalStandbyCacheableBlockCacheKeys());
        assertEquals(2192, request.getLocalStandbyBlockSize());
        assertNull(request.getInputIds());
        verifyNoInteractions(configResolver, executor, localStandbyHashService);
    }

    @Test
    void reusesSglangEagleHashAndFullBigramPagesForLocalStandby() {
        CacheMatchConfiguration configuration = configurationWithLocalStandby(4);
        FlexMonitor monitor = mock(FlexMonitor.class);
        BlockHashExecutor realExecutor =
                new BlockHashExecutor(
                        monitor, new SglangBlockHashStrategy(), 1, 2, 60, 4);
        LocalStandbyHashService standbyHashService =
                new LocalStandbyHashService(
                        configuration, monitor, new SglangBlockHashStrategy());
        RequestBlockHashService realService = new RequestBlockHashService(
                () -> new BlockHashConfig(4, 1),
                realExecutor,
                standbyHashService,
                configuration);
        Request request = new Request();
        request.setInputIds(new int[]{1, 2, 3, 4, 5, 6});

        try {
            realService.prepareBlockCacheKeys(contextFor(request)).block();

            assertEquals(
                    List.of(-638950109823820341L), request.getBlockCacheKeys());
            assertSame(
                    request.getBlockCacheKeys(),
                    request.getLocalStandbyBlockCacheKeys());
            assertEquals(
                    List.of(-638950109823820341L),
                    request.getLocalStandbyCacheableBlockCacheKeys());
        } finally {
            realExecutor.shutdown();
            standbyHashService.shutdown();
        }
    }

    @Test
    void rejectsEmptyKeysAndInputIds() {
        Request request = new Request();

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> service.prepareBlockCacheKeys(contextFor(request)).block());

        assertEquals(
                "block_cache_keys and input_ids must not both be empty",
                error.getMessage());
    }

    private BalanceContext contextFor(Request request) {
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        return context;
    }

    private CacheMatchConfiguration configurationWithLocalStandby(long blockSize) {
        KvcmConfig kvcmTopology = new KvcmConfig();
        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcmTopology);
        ModelMetaConfig config = new ModelMetaConfig();
        config.putServiceRoute(route.getServiceId(), route);
        return kvcm(config,
                runtime -> runtime.getLocalStandby().setBlockSize(blockSize));
    }
}
