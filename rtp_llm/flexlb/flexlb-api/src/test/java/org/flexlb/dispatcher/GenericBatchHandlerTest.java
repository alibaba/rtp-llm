package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.mock.web.reactive.function.server.MockServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class GenericBatchHandlerTest {

    private final ObjectMapper mapper = new ObjectMapper();
    private final BatchEndpointSpec spec = new BatchEndpointSpec(
            "/batch_infer", "prompt_batch", "response_batch", FailedItemFactory.NULL, null);

    @Test
    void singleChunkFansOutOfOneAndReturns200() {
        FanoutService fanout = mock(FanoutService.class);
        ObjectNode feResp = mapper.createObjectNode();
        ArrayNode rb = feResp.putArray("response_batch");
        rb.addObject().put("response", "r0");
        rb.addObject().put("response", "r1");
        when(fanout.dispatchChunks(eq("/batch_infer"), anyList(), eq(spec)))
                .thenReturn(Mono.just(List.of(SubBatchResult.ok(feResp, 2, 0))));

        GenericBatchHandler handler = new GenericBatchHandler(fanout, mapper, /*K=*/5);
        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("a").add("b");

        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> assertEquals(HttpStatus.OK, resp.statusCode()))
                .verifyComplete();
        verify(fanout).dispatchChunks(eq("/batch_infer"), argThat(list -> list.size() == 1), eq(spec));
    }

    @Test
    void multiChunkSplitsAndMergesReturning200() {
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatchChunks(eq("/batch_infer"), anyList(), eq(spec))).thenAnswer(inv -> {
            List<ObjectNode> bodies = inv.getArgument(1);
            List<SubBatchResult> subs = new ArrayList<>();
            int start = 0;
            for (ObjectNode b : bodies) {
                int sz = b.get("prompt_batch").size();
                ObjectNode chunkResp = mapper.createObjectNode();
                ArrayNode arr = chunkResp.putArray("response_batch");
                for (int i = 0; i < sz; i++) {
                    arr.add("r" + (start + i));
                }
                subs.add(SubBatchResult.ok(chunkResp, sz, start));
                start += sz;
            }
            return Mono.just(subs);
        });

        GenericBatchHandler handler = new GenericBatchHandler(fanout, mapper, /*K=*/2);
        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("a").add("b").add("c");

        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> assertEquals(HttpStatus.OK, resp.statusCode()))
                .verifyComplete();
        verify(fanout).dispatchChunks(eq("/batch_infer"), argThat(list -> list.size() == 2), eq(spec));
    }

    @Test
    void allFailedReturns500() {
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatchChunks(any(), anyList(), eq(spec))).thenReturn(
                Mono.just(List.of(
                        SubBatchResult.failed(2, 0, "fe_down"),
                        SubBatchResult.failed(2, 2, "fe_down"))));

        GenericBatchHandler handler = new GenericBatchHandler(fanout, mapper, /*K=*/2);
        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("a").add("b").add("c").add("d");

        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, resp.statusCode()))
                .verifyComplete();
    }

    @Test
    void nonObjectBodyReturns400() {
        FanoutService fanout = mock(FanoutService.class);
        GenericBatchHandler handler = new GenericBatchHandler(fanout, mapper, /*K=*/2);

        JsonNode arrayBody = mapper.createArrayNode().add("a").add("b");
        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(arrayBody));

        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> assertEquals(HttpStatus.BAD_REQUEST, resp.statusCode()))
                .verifyComplete();
        verifyNoInteractions(fanout);
    }

    @Test
    void missingArrayFieldReturns400() {
        FanoutService fanout = mock(FanoutService.class);
        GenericBatchHandler handler = new GenericBatchHandler(fanout, mapper, /*K=*/2);

        ObjectNode body = mapper.createObjectNode();
        body.put("unrelated", "data");
        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> assertEquals(HttpStatus.BAD_REQUEST, resp.statusCode()))
                .verifyComplete();
        verifyNoInteractions(fanout);
    }

    @Test
    void nonArrayRequestFieldReturns400() {
        FanoutService fanout = mock(FanoutService.class);
        GenericBatchHandler handler = new GenericBatchHandler(fanout, mapper, /*K=*/2);

        ObjectNode body = mapper.createObjectNode();
        body.put("prompt_batch", "not-an-array");
        MockServerRequest req = MockServerRequest.builder()
                .method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/batch_infer"))
                .body(Mono.just(body));

        StepVerifier.create(handler.handle(req, spec))
                .assertNext(resp -> assertEquals(HttpStatus.BAD_REQUEST, resp.statusCode()))
                .verifyComplete();
        verifyNoInteractions(fanout);
    }

}
