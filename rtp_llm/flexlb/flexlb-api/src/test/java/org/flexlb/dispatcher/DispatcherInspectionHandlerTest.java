package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.BatchScheduleCoordinator;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.core.io.buffer.DataBufferLimitException;
import org.springframework.http.HttpMethod;
import org.springframework.mock.web.reactive.function.server.MockServerRequest;
import org.springframework.web.reactive.function.server.EntityResponse;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@Timeout(30)
class DispatcherInspectionHandlerTest {
    private final DispatchConfig cfg = new DispatchConfig();
    private final FlexlbConfig lbConfig = new FlexlbConfig();
    private final BatchScheduleCoordinator coordinator = mock(BatchScheduleCoordinator.class);
    private final DispatcherFePoolRefresher refresher = mock(DispatcherFePoolRefresher.class);
    private final FeHealthChecker health = mock(FeHealthChecker.class);

    @BeforeEach
    void setUp() {
        cfg.setFePoolServiceId("test.fe.publish");
        cfg.setSubBatch("size:2");
        when(refresher.source()).thenReturn(List::of);
    }

    @ParameterizedTest
    @CsvSource({"3,false,test.fe.publish", "3,true,test.fe.publish", "0,false,very.specific.weird.name.publish"})
    void snapshotPreservesDiscoveryOrderAndHealth(int count, boolean mixed, String service) {
        cfg.setFePoolServiceId(service);
        List<String> urls = List.of("http://10.0.0.1:23840", "http://10.0.0.2:23840", "http://10.0.0.3:23840")
                .subList(0, count);
        when(refresher.source()).thenReturn(() -> urls);
        for (int i = 0; i < count; i++) {
            when(health.isAlive(urls.get(i))).thenReturn(!mixed || i != 1);
            when(health.consecFails(urls.get(i))).thenReturn(mixed ? (i + 1) % 3 : 0);
        }
        ServerResponse response = handler(Schedulers.immediate()).snapshot(MockServerRequest.builder()
                .method(HttpMethod.GET).uri(URI.create("http://x/dispatcher/_snapshot")).build()).block();
        JSONObject pool = body(response, 200).getJSONObject("fePool");
        assertEquals(service, pool.getString("serviceId"));
        assertEquals(count, pool.getInteger("size"));
        JSONArray hosts = pool.getJSONArray("hosts");
        assertEquals(count, hosts.size());
        for (int i = 0; i < count; i++) {
            assertEquals(urls.get(i), hosts.getJSONObject(i).getString("url"));
            assertEquals(!mixed || i != 1, hosts.getJSONObject(i).getBoolean("alive"));
            assertEquals(mixed ? (i + 1) % 3 : 0, hosts.getJSONObject(i).getInteger("consecFails"));
        }
        verifyNoInteractions(coordinator);
    }

    @ParameterizedTest
    @CsvSource(nullValues = "absent", value = {
            "false,true,BATCH_INFER,3,false", "true,true,BATCH_INFER,3,false",
            "true,false,BATCH_INFER,3,false", "true,absent,BATCH_INFER,3,false",
            "true,true,EMBEDDING,3,false", "true,true,BATCH_INFER,0,false",
            "true,true,BATCH_INFER,3,true"})
    void dryRunAssignsOnlyWhenExplicitlyRequestedAndSupported(boolean configured, String requested,
                                                              BatchEndpointSpec spec, int size, boolean policyActive) {
        cfg.setPreAssignBe(configured);
        if (policyActive) {
            TrafficPolicyConfig policy = new TrafficPolicyConfig();
            TrafficPolicyConfig.Target target = new TrafficPolicyConfig.Target();
            target.setGroup("tenant-a");
            target.setWeight(1);
            policy.setDefaultTargets(List.of(target));
            lbConfig.getRouter().setGroupSelector(policy);
        }
        when(coordinator.schedule(any())).thenReturn(Mono.just(BatchScheduleResponse.success(List.of(
                new BatchScheduleTarget("10.0.0.1", 23840, 23841, RoleType.PDFUSION),
                new BatchScheduleTarget("10.0.0.2", 23840, 23841, RoleType.PDFUSION)))));
        JSONArray items = new JSONArray();
        for (int i = 0; i < size; i++) {
            items.add("prompt-" + i);
        }
        JSONObject out = invoke(spec.getPath(), JSONObject.of(spec.getRequestArrayField(), items).toJSONString(), requested, 200);
        boolean effective = Boolean.parseBoolean(requested) && spec.isPreAssignable() && !policyActive;
        boolean allocated = effective && size > 0;
        assertEquals(configured, out.getBoolean("preAssignConfigDefault"));
        assertEquals(spec.isPreAssignable(), out.getBoolean("preAssignSupported"));
        assertEquals(effective, out.getBoolean("preAssignEffective"));
        assertEquals(size == 0 ? 0 : 2, out.getInteger("chunkCount"));
        assertEquals(allocated ? 2 : 0, out.getJSONArray("preAssignTargets").size());
        JSONArray chunks = out.getJSONArray("chunks");
        assertEquals(size == 0 ? 0 : 2, chunks.size());
        for (int i = 0; i < chunks.size(); i++) {
            JSONObject config = chunks.getJSONObject(i).getJSONObject("generate_config");
            if (spec.isPreAssignable()) {
                assertEquals(!policyActive, config.getBoolean("force_batch"));
                assertEquals(allocated, config.containsKey("role_addrs"));
                if (allocated) {
                    String ip = "10.0.0." + (i + 1);
                    assertEquals(ip, config.getJSONArray("role_addrs").getJSONObject(0).getString("ip"));
                    assertEquals(ip, out.getJSONArray("preAssignTargets").getJSONObject(i).getString("ip"));
                }
            }
        }
        if (allocated) {
            verify(coordinator).schedule(argThat(r -> r.getBatchCount() == 2 && r.isAssignBe() && !r.isAssignFe()));
        } else {
            verifyNoInteractions(coordinator);
        }
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', quoteCharacter = '~', textBlock = """
            /batch_infer | {"not_prompt_batch":"x"} | 0
            /batch_infer | {"prompt_batch":["a","b"],"images":[["http://x/0.png"],["http://x/1.png"]]} | 2
            /v1/embeddings | {"input":[{"type":"text","text":"hi"}]} | 1
            """)
    void wholeBodyInputsReportPassthrough(String path, String json, int items) {
        JSONObject out = invoke(path, json, "true", 200);
        assertEquals("passthrough", out.getString("disposition"));
        assertEquals(0, out.getInteger("chunkCount"));
        assertEquals(items, out.getInteger("totalItems"));
        verifyNoInteractions(coordinator);
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', quoteCharacter = '~', textBlock = """
            ["a"] | JSON object
            {"prompt_batch":["a"],"generate_config":"oops"} | generate_config
            {"prompt_batch":["a"],"images":[["https://example/image.png"]],"generate_config":{"role_addrs":[]}} | role_addrs
            """)
    void invalidRequestsNeverAllocate(String json, String reason) {
        JSONObject out = invoke("/batch_infer", json, "true", 400);
        assertEquals("invalid_inspection_request", out.getString("error"));
        assertTrue(out.getString("message").contains(reason));
        verifyNoInteractions(coordinator);
    }

    @Test
    void emptyBodyAndUnknownPathHaveExplicitErrors() {
        ServerResponse empty = handler(Schedulers.immediate()).dryRun(request("/batch_infer", null, Mono.empty())).block();
        assertEquals("invalid_inspection_request", body(empty, 400).getString("error"));
        JSONObject unknown = invoke("/totally_made_up", "{}", null, 400);
        assertEquals("invalid_inspection_request", unknown.getString("error"));
        assertTrue(unknown.getString("message").contains("/batch_infer"));
        assertTrue(unknown.getString("message").contains("/v1/embeddings"));
        verifyNoInteractions(coordinator);
    }

    @Test
    void rerankerRewritesChildSortingAndTopK() {
        JSONObject out = invoke("/v1/reranker", """
                {"query":"cape pants","documents":["d0","d1","d2"],"sorted":true,"top_k":2}
                """, null, 200);
        assertEquals(2, out.getInteger("chunkCount"));
        for (Object value : out.getJSONArray("chunks")) {
            JSONObject chunk = (JSONObject) value;
            assertFalse(chunk.getBoolean("sorted"));
            assertFalse(chunk.containsKey("top_k"));
        }
        verifyNoInteractions(coordinator);
    }

    @ParameterizedTest
    @CsvSource({"INVALID_REQUEST,400", "NO_AVAILABLE_WORKER,503"})
    void allocationFailuresPreserveTheirStatus(StrategyErrorType error, int status) {
        when(coordinator.schedule(any())).thenReturn(Mono.just(BatchScheduleResponse.error(error, "unavailable")));
        assertEquals("batch_schedule_failed", invoke("/batch_infer", "{\"prompt_batch\":[\"a\",\"b\"]}", "true", status).getString("error"));
    }

    @Test
    void unexpectedErrorsDoNotLeakDetails() {
        when(coordinator.schedule(any())).thenReturn(Mono.error(new RuntimeException("internal coordinator details")));
        JSONObject out = invoke("/batch_infer", "{\"prompt_batch\":[\"a\"]}", "true", 500);
        assertEquals("dryrun_internal_error", out.getString("error"));
        assertFalse(out.getString("message").contains("internal coordinator details"));
    }

    @Test
    void requestAndResponseLimitsAreEnforcedBeforeAllocation() {
        cfg.setSubBatch("size:1");
        lbConfig.getRouter().setBatchScheduleMaxCount(2);
        String json = JSONObject.of("model", "x".repeat(600), "prompt_batch", JSONArray.of("a", "b", "c")).toJSONString();
        assertEquals("too_many_sub_batches", invoke("/batch_infer", json, "true", 413).getString("error"));
        lbConfig.getRouter().setBatchScheduleMaxCount(1000);
        cfg.setMaxDryRunResponseBytes(1500);
        assertEquals("dryrun_response_too_large", invoke("/batch_infer", json, "true", 413).getString("error"));
        cfg.setMaxDryRunResponseBytes(1024);
        assertEquals(1, invoke("/batch_infer", "{\"prompt_batch\":[\"a\"]}", null, 200).getInteger("chunkCount"));
        ServerResponse oversized = handler(Schedulers.immediate()).dryRun(request("/batch_infer", null,
                Mono.error(new DataBufferLimitException("too large")))).block();
        assertEquals("request_body_too_large", body(oversized, 413).getString("error"));
        verifyNoInteractions(coordinator);
    }

    @Test
    void jsonWorkUsesTheCpuScheduler() {
        Scheduler scheduler = Schedulers.newSingle("dryrun-json-test");
        try {
            AtomicReference<String> thread = new AtomicReference<>();
            ServerResponse response = handler(scheduler).dryRun(request("/batch_infer", null,
                            Mono.just("not-json".getBytes(StandardCharsets.UTF_8))))
                    .doOnNext(ignored -> thread.set(Thread.currentThread().getName())).block();
            assertEquals("invalid_inspection_request", body(response, 400).getString("error"));
            assertTrue(thread.get().startsWith("dryrun-json-test"));
        } finally {
            scheduler.dispose();
        }
    }

    private DispatcherInspectionHandler handler(Scheduler scheduler) {
        cfg.setSubBatchSpec(SubBatchSpec.parse(cfg.getSubBatch()));
        return new DispatcherInspectionHandler(cfg, refresher, health, coordinator,
                DispatcherTestSupport.configService(lbConfig), scheduler);
    }

    private JSONObject invoke(String path, String json, String preassign, int status) {
        return body(handler(Schedulers.immediate()).dryRun(request(path, preassign,
                Mono.just(json.getBytes(StandardCharsets.UTF_8)))).block(), status);
    }

    private static MockServerRequest request(String path, String preassign, Mono<byte[]> bytes) {
        MockServerRequest.Builder builder = MockServerRequest.builder().method(HttpMethod.POST)
                .uri(URI.create("http://x/dispatcher/_dryrun" + path));
        if (preassign != null) {
            builder.queryParam("pre_assign", preassign);
        }
        return builder.body(bytes);
    }

    private static JSONObject body(ServerResponse response, int status) {
        assertEquals(status, response.rawStatusCode());
        return JSON.parseObject((byte[]) ((EntityResponse<?>) response).entity());
    }
}
