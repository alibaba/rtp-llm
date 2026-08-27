package org.flexlb.cache.hash;

import org.flexlb.cache.domain.BlockHashCalculationResult;
import org.flexlb.cache.domain.BlockHashConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class RequestBlockHashServiceTest {

    private final BlockHashConfigResolver configResolver = mock(BlockHashConfigResolver.class);
    private final BlockHashExecutor executor = mock(BlockHashExecutor.class);
    private final RequestBlockHashService service =
            new RequestBlockHashService(configResolver, executor);

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
        verifyNoInteractions(configResolver, executor);
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
}
