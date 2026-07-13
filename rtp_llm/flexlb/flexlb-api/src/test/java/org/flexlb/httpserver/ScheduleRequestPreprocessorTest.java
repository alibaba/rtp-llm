package org.flexlb.httpserver;

import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Test;

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
    private final ScheduleRequestPreprocessor preprocessor =
            new ScheduleRequestPreprocessor(blockSizeResolver);

    @Test
    void prefersProvidedBlockCacheKeys() {
        Request request = new Request();
        List<Long> providedKeys = new ArrayList<>(List.of(11L, 22L));
        request.setBlockCacheKeys(providedKeys);
        request.setInputIds(List.of(1L, 2L, 3L, 4L));

        preprocessor.prepare(request);

        assertSame(providedKeys, request.getBlockCacheKeys());
        assertNull(request.getInputIds());
        verifyNoInteractions(blockSizeResolver);
    }

    @Test
    void calculatesKeysFromInputIdsWhenProvidedKeysAreEmpty() {
        Request request = new Request();
        request.setBlockCacheKeys(List.of());
        request.setInputIds(List.of(1L, 2L, 3L, 4L, 5L));
        when(blockSizeResolver.resolve()).thenReturn(4L);

        preprocessor.prepare(request);

        assertEquals(List.of(2164874634404590027L), request.getBlockCacheKeys());
        assertNull(request.getInputIds());
    }

    @Test
    void acceptsInputIdsWithoutACompleteBlock() {
        Request request = new Request();
        request.setInputIds(List.of(1L, 2L));
        request.setBlockSize(4);

        preprocessor.prepare(request);

        assertEquals(List.of(), request.getBlockCacheKeys());
        assertNull(request.getInputIds());
        verifyNoInteractions(blockSizeResolver);
    }

    @Test
    void rejectsRequestWhenBothInputsAreEmpty() {
        Request request = new Request();

        assertThrows(IllegalArgumentException.class, () -> preprocessor.prepare(request));
    }

    @Test
    void failsWhenWorkerBlockSizeIsUnavailable() {
        Request request = new Request();
        request.setInputIds(List.of(1L));
        when(blockSizeResolver.resolve()).thenThrow(
                new IllegalStateException("block_size is unavailable"));

        assertThrows(IllegalStateException.class, () -> preprocessor.prepare(request));
    }
}
