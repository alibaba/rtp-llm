package org.flexlb.mock.engine.service;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.mock.engine.cache.KvCacheManager;
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.execution.EngineLoop;
import org.flexlb.mock.engine.scheduler.AbstractRtpScheduler;
import org.flexlb.mock.engine.scheduler.LLMMockEngineTest;
import org.flexlb.mock.engine.scheduler.TaskHistory;
import org.junit.jupiter.api.Test;

/**
 * Checks the metrics emitted by RTP request handling.
 */
class RtpRpcMetricsTest {
    @Test void statusTimingsAreRecordedBeforeCallingTheTransportObserver() throws Exception {
        var config = RtpConfig.load(new ObjectMapper().valueToTree(LLMMockEngineTest.configuration("prefill")));
        var lock = new Object(); var history = new TaskHistory();
        try (var cache = new KvCacheManager(config.blockSize(), config.deviceBlocks(), 0);
             var engine = new EngineLoop(lock, AbstractRtpScheduler.factory(config), history, cache,
                 config.compute(), config.role(), config.id());
             var service = new RtpRpcService(RtpRpcService.Config.from(config), cache, lock, history, engine, false)) {
            service.getCacheStatus(CacheVersionPB.getDefaultInstance(), observer(service, "cache_status"));
            service.getWorkerStatus(StatusVersionPB.getDefaultInstance(), observer(service, "worker_status"));
        }
    }
    private static <T> StreamObserver<T> observer(RtpRpcService service, String kind) {
        return new StreamObserver<>() {
            @Override public void onNext(T value) {
                var samples = service.collectMetrics().stream().flatMap(family -> family.samples.stream()).toList();
                String metric = "rtp_llm_rpc_" + kind + "_total_rt_us_count";
                assertEquals(1, samples.stream().filter(sample -> sample.name.equals(metric)).findFirst().orElseThrow().value);
            }
            @Override public void onError(Throwable error) { fail(error); }
            @Override public void onCompleted() { }
        };
    }
}
