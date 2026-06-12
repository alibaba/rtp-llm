package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.resources.ConnectionProvider;
import reactor.test.StepVerifier;

@Timeout(30)
class FeClientTest {

    private MockWebServer server;
    private ConnectionProvider connectionProvider;

    @BeforeEach
    void start() throws Exception {
        server = new MockWebServer();
        server.start();
        connectionProvider = ConnectionProvider.builder("test").build();
    }

    @AfterEach
    void stop() throws Exception {
        server.shutdown();
        connectionProvider.disposeLater().block(java.time.Duration.ofSeconds(5));
    }

    @Test
    void postsToBatchInferAndReturnsResponseBytes() throws Exception {
        server.enqueue(new MockResponse()
                .setHeader("Content-Type", "application/json")
                .setBody("{\"response_batch\":[{\"response\":\"ok\"}]}"));
        DispatchConfig cfg = new DispatchConfig();
        cfg.setBatchTimeoutMs(5000);
        FeClient client = new FeClient(
                WebClient.builder(), connectionProvider, cfg);

        JSONObject body = new JSONObject();
        body.put("prompt_batch", JSONArray.of("hi"));

        String base = "http://" + server.getHostName() + ":" + server.getPort();
        StepVerifier.create(client.postBytes(base, "/batch_infer", JSON.toJSONBytes(body)))
                .assertNext(bytes -> {
                    JSONObject parsed = JSON.parseObject(bytes);
                    Assertions.assertEquals("ok",
                            parsed.getJSONArray("response_batch").getJSONObject(0).getString("response"));
                })
                .verifyComplete();

        RecordedRequest rec = server.takeRequest(5, java.util.concurrent.TimeUnit.SECONDS);
        Assertions.assertNotNull(rec, "FE never received the request within 5s");
        Assertions.assertEquals("/batch_infer", rec.getPath());
        Assertions.assertEquals("POST", rec.getMethod());
    }

    @Test
    void feSlowerThanBatchTimeoutFailsTheCall() {
        // batchTimeoutMs is the header-wait budget: an FE that has not started responding
        // within it must fail the sub-call (the chunk then degrades to SubBatchResult.failed
        // upstream) instead of hanging the whole fanout.
        server.enqueue(new MockResponse()
                .setHeadersDelay(2, java.util.concurrent.TimeUnit.SECONDS)
                .setHeader("Content-Type", "application/json")
                .setBody("{\"response_batch\":[]}"));
        DispatchConfig cfg = new DispatchConfig();
        cfg.setBatchTimeoutMs(300);
        FeClient client = new FeClient(WebClient.builder(), connectionProvider, cfg);

        String base = "http://" + server.getHostName() + ":" + server.getPort();
        StepVerifier.create(client.postBytes(base, "/batch_infer", "{}".getBytes()))
                .expectError()
                .verify(java.time.Duration.ofSeconds(5));
    }
}
