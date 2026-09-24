package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpHeaders;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;
import reactor.test.StepVerifier;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class FanoutServiceTest {
    @Test
    void admits64ChunksBeforeWaitingAndPreservesOrderAcrossQueuedWork() {
        FeClient client = mock(FeClient.class);
        List<Sinks.One<byte[]>> pending = new ArrayList<>();
        when(client.postBytes(any(), any(), any(), any(), any(), any())).thenAnswer(call -> {
            Sinks.One<byte[]> response = Sinks.one();
            pending.add(response);
            return response.asMono();
        });
        FanoutService fanout = new FanoutService(client, mock(DispatcherMetricsReporter.class), Schedulers.immediate());
        List<JSONObject> chunks = IntStream.range(0, 65)
                .mapToObj(i -> JSONObject.of("prompt_batch", JSONArray.of(i))).toList();
        byte[] response = "{\"response_batch\":[\"ok\"]}".getBytes(StandardCharsets.UTF_8);

        StepVerifier.create(fanout.dispatchChunks("/batch_infer", chunks,
                        Collections.nCopies(65, "http://fe:80"), BatchEndpointSpec.BATCH_INFER,
                        HttpHeaders.EMPTY, null))
                .then(() -> {
                    assertEquals(64, pending.size());
                    pending.getFirst().tryEmitValue(response);
                    assertEquals(65, pending.size());
                    for (int i = 64; i >= 1; i--) {
                        pending.get(i).tryEmitValue(response);
                    }
                })
                .assertNext(results -> {
                    assertEquals(65, results.size());
                    assertEquals(IntStream.range(0, 65).boxed().toList(),
                            results.stream().map(SubBatchResult::startIndex).toList());
                })
                .expectComplete()
                .verify(Duration.ofSeconds(5));
    }
}
