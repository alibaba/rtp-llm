package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;

import java.util.concurrent.Executors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class RequestRoutePlanningTest {
    @Test
    void failedInputBuildCanRetryAndConcurrentReadersShareTheCompletedInput() throws Exception {
        var route = mock(RequestRoute.class);
        doCallRealMethod().when(route).planningInput();
        when(route.seqLen())
                .thenThrow(new IllegalStateException("prediction input unavailable"))
                .thenReturn(10L);
        assertThrows(IllegalStateException.class, route::planningInput);
        try (var executor = Executors.newFixedThreadPool(4)) {
            var reads =
                    IntStream.range(0, 16)
                            .mapToObj(i -> executor.submit(route::planningInput))
                            .toList();
            var expected = reads.getFirst().get();
            for (var read : reads) assertSame(expected, read.get());
            assertEquals(10L, expected.seqLen());
            verify(route, times(2)).seqLen();
        }
    }
}
