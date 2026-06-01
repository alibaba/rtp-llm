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
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.resources.ConnectionProvider;
import reactor.test.StepVerifier;

class FeClientTest {

    private MockWebServer server;

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
    void postsToBatchInferAndReturnsResponseBytes() throws Exception {
        server.enqueue(new MockResponse()
                .setHeader("Content-Type", "application/json")
                .setBody("{\"response_batch\":[{\"response\":\"ok\"}]}"));
        DispatchConfig cfg = new DispatchConfig();
        cfg.setBatchTimeoutMs(5000);
        FeClient client = new FeClient(
                WebClient.builder(), ConnectionProvider.builder("test").build(), cfg);

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

        RecordedRequest rec = server.takeRequest();
        Assertions.assertEquals("/batch_infer", rec.getPath());
        Assertions.assertEquals("POST", rec.getMethod());
    }
}
