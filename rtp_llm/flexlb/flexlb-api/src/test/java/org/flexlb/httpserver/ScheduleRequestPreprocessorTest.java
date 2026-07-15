package org.flexlb.httpserver;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class ScheduleRequestPreprocessorTest {

    private final WorkerBlockSizeResolver blockSizeResolver = mock(WorkerBlockSizeResolver.class);
    private final BlockHashExecutor blockHashExecutor = mock(BlockHashExecutor.class);
    private final ScheduleRequestPreprocessor preprocessor =
            new ScheduleRequestPreprocessor(blockSizeResolver, blockHashExecutor);

    @Test
    void prefersProvidedBlockCacheKeys() {
        Request request = new Request();
        List<Long> providedKeys = new ArrayList<>(List.of(11L, 22L));
        request.setBlockCacheKeys(providedKeys);
        request.setInputIds(new int[]{1, 2, 3, 4});
        BalanceContext context = contextFor(request);

        preprocessor.prepare(context).block();

        assertSame(providedKeys, request.getBlockCacheKeys());
        assertNull(request.getInputIds());
        verifyNoInteractions(blockSizeResolver);
        verifyNoInteractions(blockHashExecutor);
    }

    @Test
    void calculatesKeysFromInputIdsWhenProvidedKeysAreEmpty() {
        Request request = new Request();
        request.setBlockCacheKeys(List.of());
        request.setInputIds(new int[]{1, 2, 3, 4, 5});
        BalanceContext context = contextFor(request);
        when(blockSizeResolver.resolve()).thenReturn(4L);
        when(blockHashExecutor.calculate(request.getInputIds(), 4L))
                .thenReturn(Mono.just(new BlockHashCalculationResult(
                        List.of(2164874634404590027L), 12, 34)));

        preprocessor.prepare(context).block();

        assertEquals(List.of(2164874634404590027L), request.getBlockCacheKeys());
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
        when(blockHashExecutor.calculate(request.getInputIds(), 4L))
                .thenReturn(Mono.just(new BlockHashCalculationResult(List.of(), 5, 8)));

        preprocessor.prepare(context).block();

        assertEquals(List.of(), request.getBlockCacheKeys());
        assertNull(request.getInputIds());
        verifyNoInteractions(blockSizeResolver);
    }

    @Test
    void rejectsRequestWhenBothInputsAreEmpty() {
        Request request = new Request();

        assertThrows(IllegalArgumentException.class, () -> preprocessor.prepare(contextFor(request)).block());
    }

    @Test
    void failsWhenWorkerBlockSizeIsUnavailable() {
        Request request = new Request();
        request.setInputIds(new int[]{1});
        when(blockSizeResolver.resolve()).thenThrow(
                new IllegalStateException("block_size is unavailable"));

        assertThrows(IllegalStateException.class, () -> preprocessor.prepare(contextFor(request)).block());
    }

    private BalanceContext contextFor(Request request) {
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        return context;
    }
}
