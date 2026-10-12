package org.flexlb.mockengine;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

/** Direct-engine observation: excludes Master routing and batch queue repricing. */
class H2dVisibilityMicroTest {
    @TempDir Path dir;

    private Object field(Object object, String name) throws Exception {
        var f = object.getClass().getDeclaredField(name);
        f.setAccessible(true);
        return f.get(object);
    }

    @Test void lateSamePrefixProbeQuantifiesVisibilityAndAllocationGap() throws Exception {
        for (String policy : List.of("consume_only", "retain_memory", "publish_device")) {
            Path perf = dir.resolve(policy + ".json"), master = dir.resolve("master.json");
            Files.writeString(perf, "{\"block_size\":64,\"prefill\":{\"memory_cache\":{\"enabled\":true,\"capacity_blocks\":10}}}");
            MockMasterConfig.writeWithPrefillExpression(master, "1 + sum(computeTokens)");
            var model = MockPerformanceModel.load(perf.toString(), master.toString());
            model.memoryConsumeOnRead = !policy.equals("retain_memory");
            model.h2dPublishDevice = policy.equals("publish_device");
            model.setOverrideFixedPrefillMs(1000.0);
            try (var cluster = MockEngineTestCluster.create(model, 61500, 2, 0)) {
                var p = cluster.prefill(0);
                var mem = (MockMemoryBlockCache) field(p, "memoryCache");
                var gpu = (MockLruBlockCache) field(p, "cache");
                mem.write(List.of(1L, 2L));
                var input = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(700)
                        .addAllTokenIds(java.util.Collections.nCopies(192, 123))
                        .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder().setMaxNewTokens(1)
                                .setUniqueKey("{\"input_len\":192,\"output_len\":1,\"block_cache_keys\":[1,2,3]}"))
                        .build();
                var done = new CountDownLatch(1);
                var error = new AtomicReference<Throwable>();
                long start = System.nanoTime();
                p.generateStreamCall(input, new io.grpc.stub.StreamObserver<EngineRpcService.GenerateOutputsPB>() {
                    public void onNext(EngineRpcService.GenerateOutputsPB x) {
                        if (x.hasErrorInfo()) error.set(new AssertionError(x.getErrorInfo().toString()));
                    }
                    public void onError(Throwable t) { error.set(t); done.countDown(); }
                    public void onCompleted() { done.countDown(); }
                });
                Thread.sleep(100);
                assertEquals(1, done.getCount(), "probe must fall inside producer forward");
                int deviceHits = gpu.prefixHitBlocks(List.of(1L, 2L, 3L));
                var probe = p.matchPrefillMemory(new MockPerformanceModel.RequestShape(input, 192, 1,
                        List.of(1L, 2L, 3L), deviceHits * 64L, deviceHits, false));
                int expected = policy.equals("consume_only") ? 0 : 2;
                assertEquals(expected, probe.hitBlocks());
                // Physical acquisition reuses only indexed GPU keys, not host keys.
                int heldBefore = gpu.heldBlocks();
                var peer = gpu.acquire(3, List.of(1L, 2L, 3L).subList(0, deviceHits));
                assertNotNull(peer);
                int extra = gpu.heldBlocks() - heldBefore;
                assertEquals(policy.equals("publish_device") ? 1 : 3, extra);
                var otherGpu = (MockLruBlockCache) field(cluster.prefill(1), "cache");
                var otherProbe = cluster.prefill(1).matchPrefillMemory(new MockPerformanceModel.RequestShape(
                        input, 192, 1, List.of(1L, 2L, 3L), 0, 0, false));
                assertEquals(0, otherGpu.prefixHitBlocks(List.of(1L, 2L, 3L)));
                assertEquals(0, otherProbe.hitBlocks(), "publication is local to the producer P");
                assertTrue(done.await(3, TimeUnit.SECONDS));
                assertNull(error.get());
                assertEquals(3, gpu.prefixHitBlocks(List.of(1L, 2L, 3L)), "all policies converge after producer completion");
                gpu.release(peer);
                assertEquals(0, gpu.heldBlocks());
                assertEquals(0, gpu.referencedKeyBlocks());
                System.out.printf("H2D_MICRO policy=%s probe_ms=100 prefix_hits=%d memory_hits=%d device_hits=%d recompute_tokens=%d extra_private_blocks=%d other_p_hits=%d after_finish_hits=3 held=0 refs=0 producer_ms=%.1f%n",
                        policy, probe.hitBlocks(), probe.memoryHitBlocks(), deviceHits, 192-probe.hitTokens(),
                        extra, otherProbe.hitBlocks(), (System.nanoTime()-start)/1e6);
            }
        }
    }
}
