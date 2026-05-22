package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class FanoutServiceTest {

    private final ObjectMapper mapper = new ObjectMapper();

    private JsonNode batchOf(String... responses) throws Exception {
        StringBuilder sb = new StringBuilder("{\"response_batch\":[");
        for (int i = 0; i < responses.length; i++) {
            if (i > 0) {
                sb.append(",");
            }
            sb.append("{\"response\":\"").append(responses[i]).append("\"}");
        }
        sb.append("]}");
        return mapper.readTree(sb.toString());
    }

    @Test
    void splitsFansOutAndMergesInOrder() throws Exception {
        FeClient feClient = mock(FeClient.class);
        FePool pool = new FePool(() -> List.of("http://a", "http://b"));
        // chunk0 -> a -> ["r0","r1"], chunk1 -> b -> ["r2"]
        when(feClient.postBatch(eq("http://a"), any())).thenReturn(Mono.just(batchOf("r0", "r1")));
        when(feClient.postBatch(eq("http://b"), any())).thenReturn(Mono.just(batchOf("r2")));

        FanoutService svc = new FanoutService(feClient, pool, mapper, 2 /*K*/);

        StepVerifier.create(svc.dispatch(List.of("p0", "p1", "p2"), null))
                .assertNext(m -> {
                    JsonNode arr = m.body().get("response_batch");
                    assertEquals(3, arr.size());
                    assertEquals("r0", arr.get(0).get("response").asText());
                    assertEquals("r2", arr.get(2).get("response").asText());
                    assertEquals(2, m.succeededChunks());
                    assertFalse(m.allFailed());
                })
                .verifyComplete();
    }

    @Test
    void failedChunkBecomesPlaceholdersNotAnError() throws Exception {
        FeClient feClient = mock(FeClient.class);
        FePool pool = new FePool(() -> List.of("http://a", "http://b"));
        when(feClient.postBatch(eq("http://a"), any())).thenReturn(Mono.just(batchOf("r0", "r1")));
        when(feClient.postBatch(eq("http://b"), any())).thenReturn(Mono.error(new RuntimeException("FE down")));

        FanoutService svc = new FanoutService(feClient, pool, mapper, 2 /*K*/);

        StepVerifier.create(svc.dispatch(List.of("p0", "p1", "p2"), null))
                .assertNext(m -> {
                    JsonNode arr = m.body().get("response_batch");
                    assertEquals(3, arr.size()); // still N, order preserved
                    assertEquals("r0", arr.get(0).get("response").asText());
                    assertTrue(arr.get(2).get("response").asText().isEmpty()); // failed chunk -> placeholder
                    assertEquals(1, m.succeededChunks());
                    assertFalse(m.allFailed());
                })
                .verifyComplete();
    }

    @Test
    void allChunksFailedReportedNotThrown() {
        FeClient feClient = mock(FeClient.class);
        FePool pool = new FePool(() -> List.of("http://a"));
        when(feClient.postBatch(anyString(), any())).thenReturn(Mono.error(new RuntimeException("FE down")));

        FanoutService svc = new FanoutService(feClient, pool, mapper, 5 /*K*/);

        StepVerifier.create(svc.dispatch(List.of("p0", "p1"), null))
                .assertNext(m -> {
                    assertTrue(m.allFailed());
                    assertEquals(2, m.body().get("response_batch").size()); // placeholders, no exception
                })
                .verifyComplete();
    }

    @Test
    void emptyFePoolFailsChunksSoftlyWithPlaceholders() {
        FeClient feClient = mock(FeClient.class);
        FePool pool = new FePool(List::of);

        FanoutService svc = new FanoutService(feClient, pool, mapper, 2 /*K*/);

        StepVerifier.create(svc.dispatch(List.of("p0", "p1", "p2"), null))
                .assertNext(m -> {
                    assertTrue(m.allFailed());
                    JsonNode arr = m.body().get("response_batch");
                    assertEquals(3, arr.size());
                    assertTrue(arr.get(0).get("response").asText().isEmpty());
                    assertTrue(arr.get(1).get("response").asText().isEmpty());
                    assertTrue(arr.get(2).get("response").asText().isEmpty());
                })
                .verifyComplete();
    }
}
