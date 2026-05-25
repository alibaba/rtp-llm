package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class FanoutServiceTest {

    private final ObjectMapper mapper = new ObjectMapper();

    private final BatchEndpointSpec spec = new BatchEndpointSpec(
            "/batch_infer", "prompt_batch", "response_batch", FailedItemFactory.NULL, null);

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

    private ObjectNode chunk(String... prompts) {
        ObjectNode body = mapper.createObjectNode();
        var arr = body.putArray("prompt_batch");
        for (String p : prompts) {
            arr.add(p);
        }
        return body;
    }

    @Test
    void fansOutChunksAndPreservesOrder() throws Exception {
        FeClient feClient = mock(FeClient.class);
        FePool pool = new FePool(() -> List.of("http://a", "http://b"));
        when(feClient.post(eq("http://a"), eq("/batch_infer"), any())).thenReturn(Mono.just(batchOf("r0", "r1")));
        when(feClient.post(eq("http://b"), eq("/batch_infer"), any())).thenReturn(Mono.just(batchOf("r2")));

        FanoutService svc = new FanoutService(feClient, pool);

        StepVerifier.create(svc.dispatchChunks("/batch_infer", List.of(chunk("p0", "p1"), chunk("p2")), spec))
                .assertNext(subs -> {
                    assertEquals(2, subs.size());
                    SubBatchResult s0 = subs.get(0);
                    assertTrue(s0.isSuccess());
                    assertEquals(0, s0.startIndex());
                    assertEquals(2, s0.chunkSize());
                    assertEquals("r0", s0.body().get("response_batch").get(0).get("response").asText());
                    SubBatchResult s1 = subs.get(1);
                    assertTrue(s1.isSuccess());
                    assertEquals(2, s1.startIndex());
                    assertEquals(1, s1.chunkSize());
                    assertEquals("r2", s1.body().get("response_batch").get(0).get("response").asText());
                })
                .verifyComplete();
    }

    @Test
    void failedChunkBecomesFailedSubResultNotAnError() throws Exception {
        FeClient feClient = mock(FeClient.class);
        FePool pool = new FePool(() -> List.of("http://a", "http://b"));
        when(feClient.post(eq("http://a"), eq("/batch_infer"), any())).thenReturn(Mono.just(batchOf("r0", "r1")));
        when(feClient.post(eq("http://b"), eq("/batch_infer"), any())).thenReturn(Mono.error(new RuntimeException("FE down")));

        FanoutService svc = new FanoutService(feClient, pool);

        StepVerifier.create(svc.dispatchChunks("/batch_infer", List.of(chunk("p0", "p1"), chunk("p2")), spec))
                .assertNext(subs -> {
                    assertEquals(2, subs.size());
                    assertTrue(subs.get(0).isSuccess());
                    assertFalse(subs.get(1).isSuccess());
                    assertEquals(2, subs.get(1).startIndex());
                    assertEquals(1, subs.get(1).chunkSize());
                    assertNotNull(subs.get(1).reason());
                    assertTrue(subs.get(1).reason().contains("RuntimeException"));
                })
                .verifyComplete();
    }

    @Test
    void allChunksFailedReportedNotThrown() {
        FeClient feClient = mock(FeClient.class);
        FePool pool = new FePool(() -> List.of("http://a"));
        when(feClient.post(anyString(), eq("/batch_infer"), any())).thenReturn(Mono.error(new RuntimeException("FE down")));

        FanoutService svc = new FanoutService(feClient, pool);

        StepVerifier.create(svc.dispatchChunks("/batch_infer", List.of(chunk("p0", "p1")), spec))
                .assertNext(subs -> {
                    assertEquals(1, subs.size());
                    assertFalse(subs.get(0).isSuccess());
                    assertEquals(0, subs.get(0).startIndex());
                    assertEquals(2, subs.get(0).chunkSize());
                    assertTrue(subs.get(0).reason().contains("RuntimeException"));
                })
                .verifyComplete();
    }

    @Test
    void emptyFePoolFailsChunksSoftly() {
        FeClient feClient = mock(FeClient.class);
        FePool pool = new FePool(List::of);

        FanoutService svc = new FanoutService(feClient, pool);

        StepVerifier.create(svc.dispatchChunks("/batch_infer", List.of(chunk("p0", "p1"), chunk("p2")), spec))
                .assertNext(subs -> {
                    assertEquals(2, subs.size());
                    assertFalse(subs.get(0).isSuccess());
                    assertFalse(subs.get(1).isSuccess());
                    assertTrue(subs.get(0).reason().contains("IllegalStateException"));
                    assertEquals(0, subs.get(0).startIndex());
                    assertEquals(2, subs.get(1).startIndex());
                })
                .verifyComplete();
    }
}
