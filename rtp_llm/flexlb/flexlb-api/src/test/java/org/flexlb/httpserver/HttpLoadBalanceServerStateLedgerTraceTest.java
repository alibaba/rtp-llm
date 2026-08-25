package org.flexlb.httpserver;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RouteService;
import org.flexlb.sync.shadow.StateShadowBridge;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.nullValue;
import static org.mockito.Mockito.mock;

/**
 * {@code GET /rtp_llm/state_ledger/trace/{requestId}} 端点 JSON 契约：
 * 双侧活跃条目故事线（相位/世代绑定/批次/时间戳）+ 双侧墓碑终局（含
 * trace 环快照）+ 非法 ID（400）+ 账本禁用（found=false+reason）+
 * 未知请求（found=false）五态。数据源为真实 StateShadowBridge
 * （shadow 开、无调度线程），经只读 {@code traceOf} 查询。
 */
class HttpLoadBalanceServerStateLedgerTraceTest {

    /** 双侧活跃条目：P 侧 QUEUED（submit 后）+ D 侧 RESERVED（reserve 后），墓碑两路均空。 */
    @Test
    void traceReturnsBothActiveEntriesWithPhaseAndBindingContract() {
        StateShadowBridge bridge = enabledBridge();
        bridge.onPrefillSubmit(100L);
        bridge.onDecodeReserve(100L, 128L, 256L, RoleType.DECODE, "10.0.0.2:9000");

        clientFor(bridge).get().uri("/rtp_llm/state_ledger/trace/100")
                .accept(MediaType.APPLICATION_JSON)
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.request_id").isEqualTo(100)
                .jsonPath("$.found").isEqualTo(true)
                // P 侧活跃条目：相位/序数/批次（submit 未派发 → batchId=-1、binding 未绑、引擎未持有）
                .jsonPath("$.prefill_active.side").isEqualTo("P")
                .jsonPath("$.prefill_active.phase").isEqualTo("QUEUED")
                .jsonPath("$.prefill_active.phase_ordinal").isEqualTo(2)
                .jsonPath("$.prefill_active.batch_id").isEqualTo(-1)
                .jsonPath("$.prefill_active.engine_owned").isEqualTo(false)
                .jsonPath("$.prefill_active.pending_cancel").isEqualTo(false)
                // 未派发未绑定：UNBOUND 哨兵三元组（-1/-1/-1，端点世代占位）
                .jsonPath("$.prefill_active.binding.endpoint_id").isEqualTo(-1)
                .jsonPath("$.prefill_active.binding.generation").isEqualTo(-1)
                .jsonPath("$.prefill_active.binding.batch_id").isEqualTo(-1)
                .jsonPath("$.prefill_active.trace").isArray()
                // D 侧活跃条目：KV 预占对（影子预占初始按 expected 保守记，seqLen 另存）
                // + 情性注册的世代绑定（endpointId/generation 数值）
                .jsonPath("$.decode_active.side").isEqualTo("D")
                .jsonPath("$.decode_active.phase").isEqualTo("RESERVED")
                .jsonPath("$.decode_active.phase_ordinal").isEqualTo(0)
                .jsonPath("$.decode_active.reserved_kv").isEqualTo(256)
                .jsonPath("$.decode_active.reserved_expected_kv").isEqualTo(256)
                .jsonPath("$.decode_active.engine_owned").isEqualTo(false)
                .jsonPath("$.decode_active.binding.endpoint_id").isNumber()
                .jsonPath("$.decode_active.binding.generation").isNumber()
                .jsonPath("$.decode_active.trace").isArray()
                // 未终局：墓碑两路为 null
                .jsonPath("$.prefill_tombstone").value(nullValue())
                .jsonPath("$.decode_tombstone").value(nullValue());
    }

    /** 双侧墓碑终局：旧路径 CANCELLED 双清后活跃条目移除、故事线保留在墓碑（保留期内可查）。 */
    @Test
    void traceReturnsBothTombstonesAfterCancelSettlesBothSides() {
        StateShadowBridge bridge = enabledBridge();
        bridge.onPrefillSubmit(200L);
        bridge.onDecodeReserve(200L, 100L, 200L, RoleType.DECODE, "10.0.0.3:9000");
        bridge.onOldTerminal(200L, "CANCELLED");

        clientFor(bridge).get().uri("/rtp_llm/state_ledger/trace/200")
                .accept(MediaType.APPLICATION_JSON)
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.request_id").isEqualTo(200)
                .jsonPath("$.found").isEqualTo(true)
                // 双清后无活跃条目
                .jsonPath("$.prefill_active").value(nullValue())
                .jsonPath("$.decode_active").value(nullValue())
                // P 侧墓碑：终态/受控原因/终局时刻/trace 环快照（人类可读相位历史）
                .jsonPath("$.prefill_tombstone.request_id").isEqualTo(200)
                .jsonPath("$.prefill_tombstone.state").isEqualTo("CANCELLED")
                .jsonPath("$.prefill_tombstone.reason").isEqualTo("CANCELLED_NEVER_ARRIVED")
                .jsonPath("$.prefill_tombstone.terminal_at_ms").isNumber()
                .jsonPath("$.prefill_tombstone.trace").isArray()
                // D 侧墓碑同终态同原因（cancel 双清两侧 reason 一致）
                .jsonPath("$.decode_tombstone.request_id").isEqualTo(200)
                .jsonPath("$.decode_tombstone.state").isEqualTo("CANCELLED")
                .jsonPath("$.decode_tombstone.reason").isEqualTo("CANCELLED_NEVER_ARRIVED")
                .jsonPath("$.decode_tombstone.terminal_at_ms").isNumber()
                .jsonPath("$.decode_tombstone.trace").isArray();
    }

    /** 非数字 requestId：400 + found=false + error 说明（端点自防御，不抛 500）。 */
    @Test
    void traceRejectsNonNumericRequestIdWithBadRequest() {
        clientFor(enabledBridge()).get().uri("/rtp_llm/state_ledger/trace/not-a-number")
                .accept(MediaType.APPLICATION_JSON)
                .exchange()
                .expectStatus().isBadRequest()
                .expectBody()
                .jsonPath("$.found").isEqualTo(false)
                .jsonPath("$.error").value(containsString("invalid requestId"));
    }

    /** 账本禁用（未装配 bridge）：200 + found=false + reason 说明（诊断端点语义，非错误）。 */
    @Test
    void traceReportsDisabledReasonWhenLedgerDisabled() {
        clientFor(null).get().uri("/rtp_llm/state_ledger/trace/1")
                .accept(MediaType.APPLICATION_JSON)
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.request_id").isEqualTo(1)
                .jsonPath("$.found").isEqualTo(false)
                .jsonPath("$.reason").value(containsString("disabled"));
    }

    /** 未知请求（无活跃条目且墓碑保留期内无记录）：200 + found=false（契约：不 404）。 */
    @Test
    void traceReportsNotFoundForUnknownRequestId() {
        clientFor(enabledBridge()).get().uri("/rtp_llm/state_ledger/trace/999999")
                .accept(MediaType.APPLICATION_JSON)
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.request_id").isEqualTo(999999)
                .jsonPath("$.found").isEqualTo(false);
    }

    private WebTestClient clientFor(StateShadowBridge bridge) {
        HttpLoadBalanceServer server = new HttpLoadBalanceServer(
                mock(LBStatusConsistencyService.class),
                mock(ConfigService.class),
                mock(RouteService.class),
                mock(EndpointRegistry.class),
                null,
                mock(ServerScheduleLatencyRecorder.class),
                bridge);
        return WebTestClient.bindToRouterFunction(server.loadBalancePrefill()).build();
    }

    /** shadow 开、无调度线程（观测 tick/janitor 由生产装配启动，端点测试只读直查）。 */
    private static StateShadowBridge enabledBridge() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbStateV2ShadowEnabled(true);
        return StateShadowBridge.create(config, null, false);
    }
}
