package org.flexlb.cache.hash;

import org.flexlb.cache.domain.BlockHashCalculationResult;
import org.flexlb.cache.domain.WorkerBlockHashConfig;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class RequestBlockHashServiceTest {

    private final BlockHashConfigResolver blockHashConfigResolver =
            mock(BlockHashConfigResolver.class);
    private final BlockHashExecutor blockHashExecutor = mock(BlockHashExecutor.class);
    private final LocalStandbyHashService localStandbyHashService =
            mock(LocalStandbyHashService.class);
    private final RequestBlockHashService requestBlockHashService =
            new RequestBlockHashService(
                    blockHashConfigResolver,
                    blockHashExecutor,
                    localStandbyHashService,
                    new CacheMatchConfiguration(new ModelMetaConfig()));

    @Test
    void prefersProvidedBlockCacheKeys() {
        Request request = new Request();
        List<Long> providedKeys = new ArrayList<>(List.of(11L, 22L));
        request.setBlockCacheKeys(providedKeys);
        request.setBlockSize(2192);
        request.setInputIds(new int[]{1, 2, 3, 4});
        BalanceContext context = contextFor(request);

        requestBlockHashService.prepareBlockCacheKeys(context).block();

        assertSame(providedKeys, request.getBlockCacheKeys());
        assertNull(request.getInputIds());
        verifyNoInteractions(blockHashConfigResolver);
        verifyNoInteractions(blockHashExecutor);
        verifyNoInteractions(localStandbyHashService);
    }

    @Test
    void rejectsProvidedKeysWithoutBlockSize() {
        Request request = new Request();
        request.setBlockCacheKeys(List.of(11L, 22L));

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> requestBlockHashService.prepareBlockCacheKeys(contextFor(request)).block());

        assertEquals(
                "block_size must be greater than 0 when block_cache_keys are provided",
                error.getMessage());
        verifyNoInteractions(blockHashConfigResolver);
        verifyNoInteractions(blockHashExecutor);
        verifyNoInteractions(localStandbyHashService);
    }

    @Test
    void calculatesKeysFromInputIdsWhenProvidedKeysAreEmpty() {
        Request request = new Request();
        request.setBlockCacheKeys(List.of());
        request.setInputIds(new int[]{1, 2, 3, 4, 5});
        BalanceContext context = contextFor(request);
        when(blockHashConfigResolver.resolve()).thenReturn(new WorkerBlockHashConfig(4L, 0));
        when(blockHashExecutor.calculate(request.getInputIds(), 4L, 0))
                .thenReturn(Mono.just(new BlockHashCalculationResult(List.of(2164874634404590027L), 12, 34)));

        requestBlockHashService.prepareBlockCacheKeys(context).block();

        assertEquals(List.of(2164874634404590027L), request.getBlockCacheKeys());
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
        when(blockHashConfigResolver.resolve()).thenReturn(new WorkerBlockHashConfig(64L, 1));
        when(blockHashExecutor.calculate(request.getInputIds(), 4L, 1))
                .thenReturn(Mono.just(new BlockHashCalculationResult(List.of(), 5, 8)));

        requestBlockHashService.prepareBlockCacheKeys(context).block();

        assertEquals(List.of(), request.getBlockCacheKeys());
        assertNull(request.getInputIds());
    }

    @Test
    void rejectsRequestWhenBothInputsAreEmpty() {
        Request request = new Request();

        assertThrows(
                IllegalArgumentException.class,
                () -> requestBlockHashService.prepareBlockCacheKeys(
                        contextFor(request)).block());
    }

    @Test
    void failsWhenWorkerBlockHashConfigIsUnavailable() {
        Request request = new Request();
        request.setInputIds(new int[]{1});
        when(blockHashConfigResolver.resolve()).thenThrow(
                new IllegalStateException("block hash configuration is unavailable"));

        assertThrows(
                IllegalStateException.class,
                () -> requestBlockHashService.prepareBlockCacheKeys(
                        contextFor(request)).block());
    }

    @Test
    void calculatesLocalStandbyKeysWithoutWaitingOnPrimaryPath() {
        RequestBlockHashService standbyService =
                new RequestBlockHashService(
                        blockHashConfigResolver,
                        blockHashExecutor,
                        localStandbyHashService,
                        configurationWithLocalStandby(4096));
        Request request = new Request();
        int[] inputIds = new int[]{1, 2, 3, 4};
        request.setInputIds(inputIds);
        BalanceContext context = contextFor(request);
        when(blockHashConfigResolver.resolve()).thenReturn(new WorkerBlockHashConfig(2192, 1));
        when(blockHashExecutor.calculate(inputIds, 2192, 1))
                .thenReturn(Mono.just(new BlockHashCalculationResult(List.of(11L, 22L), 12, 34)));

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
                blockHashConfigResolver,
                blockHashExecutor,
                localStandbyHashService,
                configurationWithLocalStandby(2192));
        Request request = new Request();
        int[] inputIds = new int[]{1, 2, 3, 4};
        request.setInputIds(inputIds);
        List<Long> calculatedKeys = List.of(11L, 22L);
        when(blockHashConfigResolver.resolve()).thenReturn(new WorkerBlockHashConfig(2192, 1));
        when(blockHashExecutor.calculate(inputIds, 2192, 1))
                .thenReturn(Mono.just(new BlockHashCalculationResult(calculatedKeys, 12, 34)));

        standbyService.prepareBlockCacheKeys(contextFor(request)).block();

        assertSame(calculatedKeys, request.getBlockCacheKeys());
        assertSame(calculatedKeys, request.getLocalStandbyBlockCacheKeys());
        assertEquals(2192, request.getLocalStandbyBlockSize());
        verify(localStandbyHashService, never()).submit(any(), any(), anyLong(), anyInt());
    }

    @Test
    void reusesSglangEagleHashAndFullBigramPagesForLocalStandby() {
        CacheMatchConfiguration configuration = configurationWithLocalStandby(4);
        FlexMonitor monitor = mock(FlexMonitor.class);
        BlockHashExecutor executor =
                new BlockHashExecutor(monitor, new SglangBlockHashStrategy(), 1, 2, 60, 4);
        LocalStandbyHashService standbyHashService =
                new LocalStandbyHashService(
                        configuration, monitor, new SglangBlockHashStrategy());
        RequestBlockHashService service = new RequestBlockHashService(
                () -> new WorkerBlockHashConfig(4, 1),
                executor,
                standbyHashService,
                configuration);
        Request request = new Request();
        request.setInputIds(new int[]{1, 2, 3, 4, 5, 6});

        try {
            service.prepareBlockCacheKeys(contextFor(request)).block();

            assertEquals(
                    List.of(-638950109823820341L),
                    request.getBlockCacheKeys());
            assertSame(
                    request.getBlockCacheKeys(), request.getLocalStandbyBlockCacheKeys());
            assertEquals(
                    List.of(-638950109823820341L),
                    request.getLocalStandbyCacheableBlockCacheKeys());
        } finally {
            executor.shutdown();
            standbyHashService.shutdown();
        }
    }

    @Test
    void usesProvidedBlockCacheKeysForLocalStandby() {
        RequestBlockHashService standbyService =
                new RequestBlockHashService(
                        blockHashConfigResolver,
                        blockHashExecutor,
                        localStandbyHashService,
                        configurationWithLocalStandby(4096));
        Request request = new Request();
        List<Long> providedKeys = List.of(11L, 22L);
        request.setBlockCacheKeys(providedKeys);
        request.setBlockSize(2192);
        request.setInputIds(new int[]{1, 2, 3, 4});

        standbyService.prepareBlockCacheKeys(contextFor(request)).block();

        assertSame(providedKeys, request.getLocalStandbyBlockCacheKeys());
        assertEquals(2192, request.getLocalStandbyBlockSize());
        assertNull(request.getInputIds());
        verifyNoInteractions(blockHashConfigResolver);
        verifyNoInteractions(blockHashExecutor);
        verifyNoInteractions(localStandbyHashService);
    }

    private CacheMatchConfiguration configurationWithLocalStandby(long blockSize) {
        LocalStandbyConfig standby = new LocalStandbyConfig();
        standby.setBlockSize(blockSize);
        KvcmConfig kvcm = new KvcmConfig();
        kvcm.setEnabled(true);
        kvcm.setLocalStandby(standby);
        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcm);
        ModelMetaConfig config = new ModelMetaConfig();
        config.putServiceRoute(route.getServiceId(), route);
        return new CacheMatchConfiguration(config);
    }

    private BalanceContext contextFor(Request request) {
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        return context;
    }
}
