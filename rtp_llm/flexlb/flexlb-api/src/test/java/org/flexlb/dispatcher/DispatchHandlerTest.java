package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.mock.web.reactive.function.server.MockServerRequest;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.same;
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
        return new MergedResponse(body, succeededChunks, totalChunks);
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
                .assertNext(r -> assertEquals(200, r.statusCode().value()))
                .verifyComplete();

        verify(fanout).dispatch(argThat(l -> l.size() == 2 && l.get(0).equals("p0")), any());
    }

    @Test
    void handleBatchReturns500WhenAllSubBatchesFailed() {
        FanoutService fanout = mock(FanoutService.class);
        when(fanout.dispatch(anyList(), any())).thenReturn(Mono.just(mergedOf(0, 2, "", "")));

        DispatchHandler handler = new DispatchHandler(fanout, mock(PassthroughClient.class), mapper);

        StepVerifier.create(handler.handleBatch(batchRequest()))
                .assertNext(r -> assertEquals(500, r.statusCode().value()))
                .verifyComplete();
    }

    @Test
    void handlePassthroughDelegatesToClient() {
        PassthroughClient passthrough = mock(PassthroughClient.class);
        ServerResponse stubResponse = mock(ServerResponse.class);
        ServerRequest request = MockServerRequest.builder()
                .method(HttpMethod.GET)
                .body(Mono.empty());
        when(passthrough.forward(same(request))).thenReturn(Mono.just(stubResponse));

        DispatchHandler handler = new DispatchHandler(mock(FanoutService.class), passthrough, mapper);

        StepVerifier.create(handler.handlePassthrough(request))
                .assertNext(r -> assertSame(stubResponse, r))
                .verifyComplete();

        verify(passthrough).forward(same(request));
    }
}
