package org.flexlb.dispatcher;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.Map;

import static org.flexlb.constant.MetricConstant.DISPATCHER_ALL_QPS;
import static org.flexlb.constant.MetricConstant.DISPATCHER_ALL_RT;
import static org.flexlb.constant.MetricConstant.DISPATCHER_BATCH_CHUNKS;
import static org.flexlb.constant.MetricConstant.DISPATCHER_BATCH_ITEMS;
import static org.flexlb.constant.MetricConstant.DISPATCHER_CHUNK_DETAIL_QPS;
import static org.flexlb.constant.MetricConstant.DISPATCHER_CHUNK_RT;
import static org.flexlb.constant.MetricConstant.DISPATCHER_FANOUT_RT;
import static org.flexlb.constant.MetricConstant.DISPATCHER_FEPOOL_ALIVE;
import static org.flexlb.constant.MetricConstant.DISPATCHER_FEPOOL_SIZE;
import static org.flexlb.constant.MetricConstant.DISPATCHER_PREASSIGN_RT;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class DispatcherMetricsReporterTest {

    private FlexMonitor monitor;
    private DispatcherMetricsReporter reporter;

    @BeforeEach
    void setUp() {
        monitor = mock(FlexMonitor.class);
        reporter = new DispatcherMetricsReporter(monitor);
    }

    @Test
    void init_registersEveryMetricWithItsType() {
        reporter.init();

        verify(monitor).register(DISPATCHER_ALL_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).register(DISPATCHER_ALL_RT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(DISPATCHER_PREASSIGN_RT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(DISPATCHER_FANOUT_RT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        verify(monitor).register(DISPATCHER_CHUNK_DETAIL_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor).register(DISPATCHER_CHUNK_RT, FlexMetricType.GAUGE);
        verify(monitor).register(DISPATCHER_BATCH_ITEMS, FlexMetricType.GAUGE);
        verify(monitor).register(DISPATCHER_BATCH_CHUNKS, FlexMetricType.GAUGE);
        verify(monitor).register(DISPATCHER_FEPOOL_SIZE, FlexMetricType.GAUGE);
        verify(monitor).register(DISPATCHER_FEPOOL_ALIVE, FlexMetricType.GAUGE);
    }

    @Test
    void reportRequest_emitsQpsAndRtWithTypePathCodeTags() {
        reporter.reportRequest("batch", "/batch_infer", 499, 502L);

        ArgumentCaptor<FlexMetricTags> qpsTags = ArgumentCaptor.forClass(FlexMetricTags.class);
        verify(monitor).report(eq(DISPATCHER_ALL_QPS), qpsTags.capture(), eq(1.0));
        Map<String, String> tags = qpsTags.getValue().getTags();
        assertEquals("batch", tags.get("type"));
        assertEquals("/batch_infer", tags.get("path"));
        assertEquals("499", tags.get("code"));
        assertEquals(3, tags.size());

        // RT carries the same tag set and the elapsed value.
        ArgumentCaptor<FlexMetricTags> rtTags = ArgumentCaptor.forClass(FlexMetricTags.class);
        verify(monitor).report(eq(DISPATCHER_ALL_RT), rtTags.capture(), eq(502.0));
        assertEquals(tags, rtTags.getValue().getTags());
    }

    @Test
    void reportBatchShape_emitsItemsAndChunksTaggedByPath() {
        reporter.reportBatchShape("/v1/embeddings", 10, 5);

        ArgumentCaptor<FlexMetricTags> itemsTags = ArgumentCaptor.forClass(FlexMetricTags.class);
        verify(monitor).report(eq(DISPATCHER_BATCH_ITEMS), itemsTags.capture(), eq(10.0));
        assertEquals("/v1/embeddings", itemsTags.getValue().getTags().get("path"));
        assertEquals(1, itemsTags.getValue().getTags().size());

        verify(monitor).report(eq(DISPATCHER_BATCH_CHUNKS), itemsTags.capture(), eq(5.0));
    }

    @Test
    void reportPreassignRt_tagsResultByWhetherTargetsReturned() {
        reporter.reportPreassignRt(7L, true);
        ArgumentCaptor<FlexMetricTags> okTags = ArgumentCaptor.forClass(FlexMetricTags.class);
        verify(monitor).report(eq(DISPATCHER_PREASSIGN_RT), okTags.capture(), eq(7.0));
        assertEquals("ok", okTags.getValue().getTags().get("result"));

        reporter.reportPreassignRt(3L, false);
        ArgumentCaptor<FlexMetricTags> emptyTags = ArgumentCaptor.forClass(FlexMetricTags.class);
        verify(monitor).report(eq(DISPATCHER_PREASSIGN_RT), emptyTags.capture(), eq(3.0));
        assertEquals("empty", emptyTags.getValue().getTags().get("result"));
    }

    @Test
    void reportFanoutRt_emitsLatency() {
        reporter.reportFanoutRt(123L);
        verify(monitor).report(eq(DISPATCHER_FANOUT_RT), eq(FlexMetricTags.of()), eq(123.0));
    }

    @Test
    void reportChunk_emitsDetailQpsAndRtWithResultAndReason() {
        reporter.reportChunk("http_5xx", 88L);

        ArgumentCaptor<FlexMetricTags> detailTags = ArgumentCaptor.forClass(FlexMetricTags.class);
        verify(monitor).report(eq(DISPATCHER_CHUNK_DETAIL_QPS), detailTags.capture(), eq(1.0));
        assertEquals("failed", detailTags.getValue().getTags().get("result"));
        assertEquals("http_5xx", detailTags.getValue().getTags().get("reason"));

        ArgumentCaptor<FlexMetricTags> rtTags = ArgumentCaptor.forClass(FlexMetricTags.class);
        verify(monitor).report(eq(DISPATCHER_CHUNK_RT), rtTags.capture(), eq(88.0));
        assertEquals("failed", rtTags.getValue().getTags().get("result"));
        assertEquals(1, rtTags.getValue().getTags().size());
    }

    @Test
    void reportFePool_emitsSizeAndAlive() {
        reporter.reportFePool(8, 6);
        verify(monitor).report(eq(DISPATCHER_FEPOOL_SIZE), eq(FlexMetricTags.of()), eq(8.0));
        verify(monitor).report(eq(DISPATCHER_FEPOOL_ALIVE), eq(FlexMetricTags.of()), eq(6.0));
    }
}
