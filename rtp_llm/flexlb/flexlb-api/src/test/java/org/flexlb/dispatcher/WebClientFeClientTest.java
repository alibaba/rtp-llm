package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.buffer.DataBufferLimitException;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.test.StepVerifier;

class WebClientFeClientTest {

    private MockWebServer server;
    private final ObjectMapper mapper = new ObjectMapper();

    @BeforeEach
    void start() throws Exception {
        server = new MockWebServer();
        server.start();
    }

    @AfterEach
    void stop() throws Exception {
        server.shutdown();
    }

    @Test
    void postsToBatchInferAndParsesResponse() throws Exception {
        server.enqueue(new MockResponse()
                .setHeader("Content-Type", "application/json")
                .setBody("{\"response_batch\":[{\"response\":\"ok\"}]}"));
        WebClientFeClient client = new WebClientFeClient(WebClient.builder(), 3000, 16 * 1024 * 1024);

        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("hi");

        String base = "http://" + server.getHostName() + ":" + server.getPort();
        StepVerifier.create(client.post(base, "/batch_infer", body))
                .assertNext(n -> Assertions.assertEquals("ok",
                        n.get("response_batch").get(0).get("response").asText()))
                .verifyComplete();

        RecordedRequest rec = server.takeRequest();
        Assertions.assertEquals("/batch_infer", rec.getPath());
        Assertions.assertEquals("POST", rec.getMethod());
    }

    @Test
    void rejectsResponseLargerThanMaxBytesCap() throws Exception {
        // Build a response_batch whose serialized body comfortably exceeds the 4 KiB cap.
        StringBuilder big = new StringBuilder("{\"response_batch\":[{\"response\":\"");
        for (int i = 0; i < 8 * 1024; i++) {
            big.append('x');
        }
        big.append("\"}]}");
        server.enqueue(new MockResponse().setHeader("Content-Type", "application/json").setBody(big.toString()));

        WebClientFeClient client = new WebClientFeClient(WebClient.builder(), 3000, /*maxResponseBytes*/ 4 * 1024);

        ObjectNode body = mapper.createObjectNode();
        body.putArray("prompt_batch").add("hi");
        String base = "http://" + server.getHostName() + ":" + server.getPort();

        StepVerifier.create(client.post(base, "/batch_infer", body))
                .expectErrorMatches(e -> e instanceof DataBufferLimitException
                        || (e.getCause() != null && e.getCause() instanceof DataBufferLimitException))
                .verify();
    }
}
