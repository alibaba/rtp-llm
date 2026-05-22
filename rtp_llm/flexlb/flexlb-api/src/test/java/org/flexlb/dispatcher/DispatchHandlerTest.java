package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.mock.web.reactive.function.server.MockServerRequest;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DispatchHandlerTest {

    private final ObjectMapper mapper = new ObjectMapper();

    private MergedResponse mergedOf(int succeededChunks, int totalChunks, String... responses) {
        ObjectNode body = mapper.createObjectNode();
        ArrayNode arr = body.putArray("response_batch");
        for (String r : responses) {
            arr.addObject().put("response", r);
        }
        return new MergedResponse(body, succeededChunks, totalChunks, responses.length);
    }

    private MockServerRequest batchRequest() {
        ObjectNode reqBody = mapper.createObjectNode();
        reqBody.putArray("prompt_batch").add("p0").add("p1");
        return MockServerRequest.builder().method(HttpMethod.POST).body(Mono.just(reqBody));
    }

    @Test
    void handleBatchExtractsPromptsAndReturns200OnPartialSuccess() {
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatch(anyList(), any())).thenReturn(Mono.just(mergedOf(1, 2, "done", "")));

        DispatchHandler handler = new DispatchHandler(fanout, mock(PassthroughClient.class), mapper);

        StepVerifier.create(handler.handleBatch(batchRequest()))
                .assertNext(r -> {
                    assert r.statusCode().value() == 200;
                })
                .verifyComplete();

        verify(fanout).dispatch(argThat(l -> l.size() == 2 && l.get(0).equals("p0")), any());
    }

    @Test
    void handleBatchReturns500WhenAllSubBatchesFailed() {
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatch(anyList(), any())).thenReturn(Mono.just(mergedOf(0, 2, "", "")));

        DispatchHandler handler = new DispatchHandler(fanout, mock(PassthroughClient.class), mapper);

        StepVerifier.create(handler.handleBatch(batchRequest()))
                .assertNext(r -> {
                    assert r.statusCode().value() == 500;
                })
                .verifyComplete();
    }
}
