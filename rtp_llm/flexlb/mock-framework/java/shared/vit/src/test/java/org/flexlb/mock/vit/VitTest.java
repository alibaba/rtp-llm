package org.flexlb.mock.vit;

import org.flexlb.mock.multimodal.MediaInput;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.IntStream;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class VitTest {
    @TempDir Path directory;
    VitConfig config(String expression,long device,long host,long hashes,int wait) throws Exception {
        var path=directory.resolve(UUID.randomUUID()+".json");
        new ObjectMapper().writeValue(path.toFile(),Map.of("id","vit-0","component_role","vit","component",Map.of("config",Map.of(
            "hidden_size",4,"mm_cache_gpu_max_bytes",device,"mm_cache_cpu_max_bytes",host,"mm_hash_key_cache_max_bytes",hashes,
            "gpu_batch_wait_ms",wait,"compute",Map.of("forward",Map.of("expression",expression))))));
        return VitConfig.load(path);
    }
    static MultimodalInputsPB input(String key) {
        return MultimodalInputsPB.newBuilder().setRequestId(1).addMultimodalInputs(MultimodalInputPB.newBuilder()
            .setMultimodalType(1).setMultimodalUrl(key).setMmPreprocessConfig(MMPreprocessConfigPB.newBuilder().setWidth(56).setHeight(28))).build();
    }
    static double sample(String text, String name) {
        return text.lines().filter(line -> line.startsWith(name + "{") && line.contains("role=\"vit\""))
            .mapToDouble(line -> Double.parseDouble(line.substring(line.lastIndexOf(' ') + 1))).findFirst().orElseThrow();
    }
    static ManagedChannel channel(VitNode node) {
        return ManagedChannelBuilder.forTarget((String)node.endpoints().get("rpc")).usePlaintext().build();
    }
    @Test void mergesRequestsIntoOneForwardAndReturnsOnlyMetadata() throws Exception {
        try(var node=VitNode.start(config("2 + batchSize",1024,1024,1024,150),0,0);
            var threads=Executors.newVirtualThreadPerTaskExecutor()) {
            var channel=channel(node);
            try {
                var gate=new CountDownLatch(1);var futures=new ArrayList<Future<MultimodalOutputPB>>();
                for(int i=0;i<4;i++) { int n=i;futures.add(threads.submit(()->{gate.await();return MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(5,TimeUnit.SECONDS).remoteMultimodalEmbedding(input("image-"+n));})); }
                gate.countDown();
                for(var future:futures) {
                    var output=future.get(5,TimeUnit.SECONDS);
                    assertEquals(List.of(2L,4L),output.getMultimodalEmbedding().getShapeList());
                    assertTrue(output.getMultimodalEmbedding().getBf16Data().isEmpty());
                    assertEquals(8,output.getMultimodalFeatureHash().getInt32Data().size());assertEquals(List.of(2),output.getSplitSizeList());
                }
                var metadata=MultimodalRpcServiceGrpc.newBlockingStub(channel).getMultimodalHashes(MultimodalHashRequestPB.newBuilder()
                    .addKeys(MediaInput.key(input("image-0").getMultimodalInputs(0))).build());
                assertTrue(metadata.getEntries(0).getHashHit());assertTrue(metadata.getEntries(0).getEmbeddingHit());
                String metrics=java.net.http.HttpClient.newHttpClient().send(java.net.http.HttpRequest.newBuilder(java.net.URI.create((String)node.endpoints().get("metrics"))).GET().build(),java.net.http.HttpResponse.BodyHandlers.ofString()).body();
                assertEquals(1, sample(metrics, "py_rtp_vit_embedding_forward_rt_count"));
                assertEquals(6, sample(metrics, "mock_vit_theoretical_forward_rt_sum"));
                for (String metric : List.of("py_rtp_vit_qps", "py_rtp_vit_success_qps", "py_rtp_vit_image_count",
                        "py_rtp_vit_embedding_length", "py_rtp_vit_embedding_queue_wait_rt", "py_rtp_vit_embedding_queue_size",
                        "py_rtp_vit_embedding_cache_tokens", "py_rtp_vit_embedding_batch_size"))
                    assertTrue(Double.isFinite(sample(metrics, metric)), metrics);
                assertFalse(metrics.contains("mock_engine_requests"));
            } finally { channel.shutdownNow().awaitTermination(2,TimeUnit.SECONDS); }
        }
    }
    @Test void logicalByteBudgetsMoveEmbeddingsIndependentlyOfHashResidency() throws Exception {
        var config=config("1",16,16,1024,0);
        try(var cache=new VitCache(config)) {
            var images=List.of(new MediaInput.Image("a",56,28,2),new MediaInput.Image("b",56,28,2),new MediaInput.Image("c",56,28,2));
            for(var image:images) { var claim=cache.acquire(image);cache.complete(claim,null); }
            assertEquals("",cache.tier("a"));assertNotNull(cache.hash("a"));
            assertEquals("cpu",cache.tier("b"));assertEquals("gpu",cache.tier("c"));
            assertFalse(cache.acquire(images.get(1)).producer());
            assertEquals("gpu",cache.tier("b"));assertEquals("cpu",cache.tier("c"));
            assertTrue(cache.snapshot().get("py_rtp_vit_embedding_cache_bytes") == 32.0);
        }
    }
    @Test void failedProducerReleasesTheClaimAndOldCompletionCannotPublishANewGeneration() throws Exception {
        try(var cache=new VitCache(config("1",16,16,1024,0))) {
            var image=new MediaInput.Image("a",56,28,2);var first=cache.acquire(image);var joined=cache.acquire(image);
            assertFalse(joined.producer());assertSame(first.result(),joined.result());
            cache.complete(first,new IllegalStateException("failed"));assertTrue(joined.result().isCompletedExceptionally());
            var fresh=cache.acquire(image);cache.complete(first,null);assertFalse(fresh.result().isDone());
            cache.complete(fresh,null);assertEquals(image,fresh.result().join());
        }
    }
    @Test void oneWaiterDeadlineDoesNotCancelAnotherWaiterOrTheSharedForward() throws Exception {
        try(var node=VitNode.start(config("150",16,16,1024,0),0,0);var threads=Executors.newVirtualThreadPerTaskExecutor()) {
            var channel=channel(node);
            try {
                var shortWait=threads.submit(()->assertThrows(StatusRuntimeException.class,()->MultimodalRpcServiceGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(40,TimeUnit.MILLISECONDS).remoteMultimodalEmbedding(input("same"))));
                var output=MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3,TimeUnit.SECONDS).remoteMultimodalEmbedding(input("same"));
                shortWait.get(3,TimeUnit.SECONDS);assertEquals(2,output.getMultimodalEmbedding().getShape(0));
            } finally { channel.shutdownNow().awaitTermination(2,TimeUnit.SECONDS); }
        }
    }
    @Test void malformedMetadataAndSpanMismatchAreRejectedBeforePAllocation() throws Exception {
        var nativeInput=GenerateInputPB.newBuilder().setRequestId(1).addAllTokenIds(List.of(10,0,0,20))
            .addAllMultimodalInputs(input("x").getMultimodalInputsList())
            .setMultimodalTokenLayout(MultimodalTokenLayoutPB.newBuilder().addSpans(MultimodalTokenSpanPB.newBuilder().setOffset(1).setLength(2))).build();
        var hashes=MediaInput.hashes(MediaInput.key(input("x").getMultimodalInputs(0)),2);
        var output=MultimodalOutputPB.newBuilder().addSplitSize(2)
            .setMultimodalEmbedding(TensorPB.newBuilder().setDataType(TensorPB.DataType.BF16).addShape(2).addShape(4))
            .setMultimodalFeatureHash(TensorPB.newBuilder().setDataType(TensorPB.DataType.INT32).addShape(2).setInt32Data(hashes)).build();
        var prepared=MediaInput.apply(nativeInput,output);assertEquals(10,prepared.getTokenIds(0));assertEquals(20,prepared.getTokenIds(3));
        assertNotEquals(nativeInput.getTokenIdsList(),prepared.getTokenIdsList());
        assertThrows(StatusRuntimeException.class,()->MediaInput.apply(nativeInput,output.toBuilder().clearSplitSize().addSplitSize(1).build()));
        assertThrows(StatusRuntimeException.class,()->MediaInput.apply(nativeInput,output.toBuilder().clearMultimodalFeatureHash().build()));
        assertThrows(StatusRuntimeException.class,()->VitGeometry.images(MultimodalInputsPB.getDefaultInstance(),14,2));
    }
    @Test void cacheKeyUsesLinuxNativeIdentityAndExcludesTimeout() {
        String[] values={"https://example.com/image.png","中文url"};
        String[] expected={"2000644620635833837","18006779371473338865"};
        for(int i=0;i<values.length;i++) assertTrue(MediaInput.key(input(values[i]).getMultimodalInputs(0)).startsWith(expected[i]+"_1_"));
        var media=input("x").getMultimodalInputs(0);
        var changed=media.toBuilder().setMmPreprocessConfig(media.getMmPreprocessConfig().toBuilder().setMmTimeoutMs(2)).build();
        assertEquals(MediaInput.key(media),MediaInput.key(changed));
    }
    @Test void queueOverflowFailsClaimsAndStopEndsEveryOutstandingWaiter() throws Exception {
        var path=directory.resolve("bounded.json");
        new ObjectMapper().writeValue(path.toFile(),Map.of("id","vit-0","component_role","vit","component",Map.of("config",Map.of(
            "gpu_max_batch_size",1,"mm_max_queue_size",1,"compute",Map.of("forward",Map.of("expression","1000"))))));
        var config=VitConfig.load(path);
        try(var cache=new VitCache(config);var scheduler=new VitScheduler(config,cache)) {
            var first=cache.acquire(new MediaInput.Image("a",28,28,1));scheduler.submit(List.of(first));
            long end=System.nanoTime()+TimeUnit.SECONDS.toNanos(2);
            while(scheduler.snapshot().get("mock_vit_running_image_count") == 0) { assertTrue(System.nanoTime()<end);Thread.sleep(2); }
            var second=cache.acquire(new MediaInput.Image("b",28,28,1));scheduler.submit(List.of(second));
            var overflow=cache.acquire(new MediaInput.Image("c",28,28,1));
            var rejected=assertThrows(StatusRuntimeException.class,()->scheduler.submit(List.of(overflow)));
            assertEquals(Status.Code.RESOURCE_EXHAUSTED,rejected.getStatus().getCode());cache.complete(overflow,rejected);
            assertTrue(overflow.result().isCompletedExceptionally());scheduler.close();
            assertThrows(ExecutionException.class,()->first.result().get(2,TimeUnit.SECONDS));
            assertThrows(ExecutionException.class,()->second.result().get(2,TimeUnit.SECONDS));
        }
    }

    @Test void hashHitDoesNotRequireEmbeddingResidencyOrAnotherForward() throws Exception {
        try(var node=VitNode.start(config("1",0,0,1024,0),0,0)) {
            var channel=channel(node);
            try {
                var rpc=MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3,TimeUnit.SECONDS);
                var request=MultimodalHashRequestPB.newBuilder().addKeys(MediaInput.key(input("x").getMultimodalInputs(0))).setInputs(input("x")).build();
                assertTrue(rpc.getMultimodalHashes(request).getEntries(0).getHashHit());
                var hit=rpc.getMultimodalHashes(request).getEntries(0);
                assertTrue(hit.getHashHit());assertFalse(hit.getEmbeddingHit());
                String metrics=java.net.http.HttpClient.newHttpClient().send(java.net.http.HttpRequest.newBuilder(java.net.URI.create((String)node.endpoints().get("metrics"))).GET().build(),java.net.http.HttpResponse.BodyHandlers.ofString()).body();
                assertEquals(1, sample(metrics, "py_rtp_vit_embedding_forward_rt_count"));
                assertEquals(0, sample(metrics, "py_rtp_vit_embedding_cache_eviction_qps"));
                rpc.remoteMultimodalEmbedding(input("x"));
                assertEquals(Status.Code.INVALID_ARGUMENT,assertThrows(StatusRuntimeException.class,()->rpc.getMultimodalHashes(request.toBuilder().clearKeys().addKeys("wrong").build())).getStatus().getCode());
            } finally { channel.shutdownNow().awaitTermination(2,TimeUnit.SECONDS); }
        }
    }

    @Test void queuedWorkWithNoConsumersIsRemovedWithoutAnotherForward() throws Exception {
        try(var cache=new VitCache(config("120",16,16,1024,0));
            var scheduler=new VitScheduler(config("120",16,16,1024,0),cache)) {
            var first=cache.acquire(new MediaInput.Image("a",28,28,1));scheduler.submit(List.of(first));
            long end=System.nanoTime()+TimeUnit.SECONDS.toNanos(2);
            while(scheduler.snapshot().get("mock_vit_running_image_count") == 0) { assertTrue(System.nanoTime()<end);Thread.sleep(2); }
            var abandoned=cache.acquire(new MediaInput.Image("b",28,28,1));scheduler.submit(List.of(abandoned));VitCache.release(abandoned);
            first.result().get(2,TimeUnit.SECONDS);
            assertThrows(ExecutionException.class,()->abandoned.result().get(2,TimeUnit.SECONDS));
            assertEquals(0, scheduler.snapshot().get("mock_vit_running_image_count"));
        }
    }

    @Test void sharedVitEngineRejectsABackendIdentity() throws Exception {
        var config=config("1",16,16,1024,0);
        var path=directory.resolve("backend.json");
        new ObjectMapper().writeValue(path.toFile(),Map.of("id","vit-0","component_role","vit",
            "component",Map.of("engine","rtp","config",Map.of("compute",Map.of("forward",Map.of("expression","1"))))));
        assertTrue(assertThrows(IllegalArgumentException.class,()->VitConfig.load(path)).getMessage().contains("shared implementation"));
        assertEquals("vit",config.effective().path("role").asText());
    }

}
