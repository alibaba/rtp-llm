package org.flexlb.mock.vit;

import org.flexlb.mock.multimodal.MediaInput;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class VitTest {
    @TempDir Path directory;
    @Test void textOnlyV41EnvelopeDoesNotRequireVisionPreparation() {
        var input = GenerateInputPB.newBuilder().addTokenIds(42)
            .setV41Inputs(V41RequestInputsPB.newBuilder().setSchemaVersion(1).addTokenTypes(0))
            .setGenerateConfig(GenerateConfigPB.newBuilder().setTimeoutMs(-1)).build();
        try (var client = new org.flexlb.mock.multimodal.MediaClient()) {
            // No V address is provided: a text-only request must not open a V RPC.
            assertSame(input, client.prepare(input, Long.MAX_VALUE));
        }
        assertTrue(MediaInput.present(input.toBuilder().setV41Inputs(input.getV41Inputs().toBuilder()
            .addImages(V41ImageInputPB.getDefaultInstance())).build()));
        assertTrue(MediaInput.present(input.toBuilder().addAllMultimodalInputs(input("image").getMultimodalInputsList()).build()));
    }
    VitConfig config(String expression,long device,long host,long hashes,int wait) throws Exception {
        return config(expression,device,host,hashes,wait,Map.of());
    }
    VitConfig config(String expression,long device,long host,long hashes,int wait,Map<String,Object> overrides) throws Exception {
        var path=directory.resolve(UUID.randomUUID()+".json");
        var settings=new HashMap<String,Object>(Map.of("mock.vit.hidden.size",4,"mock.vit.skip.download",true,
            "mm_cache_gpu_max_bytes",device,"mm_cache_cpu_max_bytes",host,"mm_hash_key_cache_max_bytes",hashes,
            "gpu_max_batch_size",8,"gpu_batch_wait_ms",wait,"mock.compute",Map.of("forward",Map.of("expression",expression))));
        settings.putAll(overrides);
        new ObjectMapper().writeValue(path.toFile(),Map.of("id","vit-0","component_role","vit","component",Map.of("config",settings)));
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
    @Test void acceptsNativeFrontendTwoHourRpcBudgetForEmbeddingAndHashRequests() throws Exception {
        try (var node=VitNode.start(config("1",1024,1024,1024,0),0,0)) {
            var channel=channel(node);
            try {
                var rpc=MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3,TimeUnit.SECONDS);
                var media=input("native-timeout").toBuilder().setTimeoutMs(7200000).build();
                var output=rpc.remoteMultimodalEmbedding(media);
                assertEquals(List.of(2),output.getSplitSizeList());
                var hashes=rpc.getMultimodalHashes(MultimodalHashRequestPB.newBuilder()
                    .setTimeoutMs(7200000).setInputs(media)
                    .addKeys(MediaInput.key(media.getMultimodalInputs(0))).build());
                assertTrue(hashes.getEntries(0).getHashHit());
            } finally { channel.shutdownNow().awaitTermination(2,TimeUnit.SECONDS); }
        }
    }
    @Test void mergesRequestsWithinImageBudgetAndReturnsOnlyMetadata() throws Exception {
        try(var node=VitNode.start(config("2 + batchSize",1024,1024,1024,150,Map.of("gpu_max_batch_images",2)),0,0);
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
                assertEquals(2, sample(metrics, "py_rtp_vit_embedding_forward_rt_count"));
                assertEquals(8, sample(metrics, "mock_vit_theoretical_forward_rt_sum"));
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
            var images=List.of(new VitGeometry.Image("a",56,28,2),new VitGeometry.Image("b",56,28,2),new VitGeometry.Image("c",56,28,2));
            for(var image:images) { var claim=cache.acquire(List.of(image.key())).getFirst();cache.publish(claim,image,null); }
            assertEquals("",cache.tier("a"));assertNotNull(cache.hash("a"));
            assertEquals("cpu",cache.tier("b"));assertEquals("gpu",cache.tier("c"));
            assertFalse(cache.acquire(List.of(images.get(1).key())).getFirst().producer());
            assertEquals("gpu",cache.tier("b"));assertEquals("cpu",cache.tier("c"));
            assertTrue(cache.snapshot().get("py_rtp_vit_embedding_cache_bytes") == 32.0);
        }
    }
    @Test void failedProducerReleasesTheClaimAndOldCompletionCannotPublishANewGeneration() throws Exception {
        try(var cache=new VitCache(config("1",16,16,1024,0))) {
            var image=new VitGeometry.Image("a",56,28,2);var first=cache.acquire(List.of(image.key())).getFirst();var joined=cache.acquire(List.of(image.key())).getFirst();
            assertFalse(joined.producer());assertSame(first.result(),joined.result());
            cache.publish(first,null,new IllegalStateException("failed"));assertTrue(joined.result().isCompletedExceptionally());
            var fresh=cache.acquire(List.of(image.key())).getFirst();cache.publish(first,image,null);assertFalse(fresh.result().isDone());
            cache.publish(fresh,image,null);assertEquals(image,fresh.result().join());
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
        var hashes=VitGeometry.hashes(MediaInput.key(input("x").getMultimodalInputs(0)),2);
        var output=MultimodalOutputPB.newBuilder().addSplitSize(2)
            .setMultimodalEmbedding(TensorPB.newBuilder().setDataType(TensorPB.DataType.BF16).addShape(2).addShape(4))
            .setMultimodalFeatureHash(TensorPB.newBuilder().setDataType(TensorPB.DataType.INT32).addShape(2).setInt32Data(hashes)).build();
        var prepared=MediaInput.apply(nativeInput,output);assertEquals(10,prepared.getTokenIds(0));assertEquals(20,prepared.getTokenIds(3));
        assertNotEquals(nativeInput.getTokenIdsList(),prepared.getTokenIdsList());
        assertThrows(StatusRuntimeException.class,()->MediaInput.apply(nativeInput,output.toBuilder().clearSplitSize().addSplitSize(1).build()));
        assertThrows(StatusRuntimeException.class,()->MediaInput.apply(nativeInput,output.toBuilder().clearMultimodalFeatureHash().build()));
        assertThrows(StatusRuntimeException.class,()->VitGeometry.validate(MultimodalInputsPB.getDefaultInstance(),true));
    }
    @Test void cacheKeyUsesLinuxNativeIdentityAndExcludesTimeout() {
        String[] values={"https://example.com/image.png","中文url"};
        String[] expected={"2000644620635833837","18006779371473338865"};
        for(int i=0;i<values.length;i++) assertTrue(MediaInput.key(input(values[i]).getMultimodalInputs(0)).startsWith(expected[i]+"_1_"));
        var media=input("x").getMultimodalInputs(0);
        var changed=media.toBuilder().setMmPreprocessConfig(media.getMmPreprocessConfig().toBuilder().setMmTimeoutMs(2)).build();
        assertEquals(MediaInput.key(media),MediaInput.key(changed));
    }
    @Test void queueOverflowRejectsWorkAndStopCompletesEveryStageWaiter() throws Exception {
        var path=directory.resolve("bounded.json");
        new ObjectMapper().writeValue(path.toFile(),Map.of("id","vit-0","component_role","vit","component",Map.of("config",Map.of(
            "mm_max_queue_size",1,"gpu_max_batch_size",1,"vit_concurrency",1,"vit_max_queue_size",0,"mock.compute",Map.of("forward",Map.of("expression","1000"))))));
        try(var scheduler=new VitScheduler(VitConfig.load(path))) {
            var first=scheduler.submit(new VitGeometry.Image("a",28,28,1));
            long end=System.nanoTime()+TimeUnit.SECONDS.toNanos(2);
            while(scheduler.snapshot().get("mock_vit_running_image_count")==0) { assertTrue(System.nanoTime()<end);Thread.sleep(2); }
            var second=scheduler.submit(new VitGeometry.Image("b",28,28,1));
            assertEquals(Status.Code.RESOURCE_EXHAUSTED,assertThrows(StatusRuntimeException.class,
                ()->scheduler.submit(new VitGeometry.Image("c",28,28,1))).getStatus().getCode());
            scheduler.close();
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

    @Test void queuedStageCanBeCancelledWithoutAnotherForward() throws Exception {
        try(var scheduler=new VitScheduler(config("120",16,16,1024,0))) {
            var first=scheduler.submit(new VitGeometry.Image("a",28,28,1));
            long end=System.nanoTime()+TimeUnit.SECONDS.toNanos(2);
            while(scheduler.snapshot().get("mock_vit_running_image_count")==0) { assertTrue(System.nanoTime()<end);Thread.sleep(2); }
            var abandoned=scheduler.submit(new VitGeometry.Image("b",28,28,1));
            assertTrue(scheduler.cancel(abandoned)); assertFalse(scheduler.cancel(first));
            first.result().get(2,TimeUnit.SECONDS);
            assertThrows(ExecutionException.class,()->abandoned.result().get(2,TimeUnit.SECONDS));
            assertEquals(0,scheduler.snapshot().get("mock_vit_running_image_count"));
        }
    }

    @Test void admissionIsReleasedAfterSharedResultPublication() throws Exception {
        var settings=config("100",1024,1024,1024,0,Map.of("vit_concurrency",1,"vit_max_queue_size",0));
        var published=new CountDownLatch(1); var finishPublication=new CountDownLatch(1);
        try(var cache=new VitCache(settings);var scheduler=new VitScheduler(settings);
            var preparation=new VitPreparation(settings,cache,scheduler,new VitMetrics())) {
            var claims=preparation.submit(input("first"));
            var notified=claims.getFirst().result().thenRun(()->{
                published.countDown();
                try { assertTrue(finishPublication.await(2,TimeUnit.SECONDS)); }
                catch(InterruptedException error) { Thread.currentThread().interrupt();throw new RuntimeException(error); }
            });
            try {
                assertTrue(published.await(2,TimeUnit.SECONDS));
                assertNotNull(cache.hash(MediaInput.key(input("first").getMultimodalInputs(0))));
                assertEquals(1,preparation.inflight());
                assertEquals(Status.Code.RESOURCE_EXHAUSTED,assertThrows(StatusRuntimeException.class,
                    ()->preparation.submit(input("second"))).getStatus().getCode());
            } finally { finishPublication.countDown(); }
            notified.get(2,TimeUnit.SECONDS); claims.forEach(VitCache::release);
            long until=System.nanoTime()+TimeUnit.SECONDS.toNanos(2);
            while(preparation.inflight()!=0) { assertTrue(System.nanoTime()<until);Thread.sleep(2); }
            var next=preparation.submit(input("second"));
            next.getFirst().result().get(2,TimeUnit.SECONDS);next.forEach(VitCache::release);
        }
    }

    @Test void sharedVitEngineRejectsABackendIdentity() throws Exception {
        var config=config("1",16,16,1024,0);
        var path=directory.resolve("backend.json");
        new ObjectMapper().writeValue(path.toFile(),Map.of("id","vit-0","component_role","vit",
            "component",Map.of("engine","rtp","config",Map.of("mock.compute",Map.of("forward",Map.of("expression","1"))))));
        assertTrue(assertThrows(IllegalArgumentException.class,()->VitConfig.load(path)).getMessage().contains("shared implementation"));
        assertEquals("vit",config.effective().path("role").asText());
    }

}
