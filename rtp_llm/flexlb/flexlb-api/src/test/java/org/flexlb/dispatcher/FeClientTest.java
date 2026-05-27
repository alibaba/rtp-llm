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
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.resources.ConnectionProvider;
import reactor.test.StepVerifier;

class FeClientTest {

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
        DispatchConfig cfg = new DispatchConfig();
        cfg.setBatchTimeoutMs(5000);
        FeClient client = new FeClient(
                WebClient.builder(), ConnectionProvider.builder("test").build(), cfg);

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

}
