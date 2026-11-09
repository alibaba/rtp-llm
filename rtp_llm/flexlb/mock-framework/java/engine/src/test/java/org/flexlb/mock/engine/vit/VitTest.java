package org.flexlb.mock.engine.vit;

import org.flexlb.mock.protocol.MediaInput;

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

/**
 * Exercises ViT admission, scheduling, cache reuse and cancellation.
 */
class VitTest {
    @TempDir Path directory;

    @Test void nativeWorkRatesCountUniqueMediaAcrossRpcCacheAndAsyncPaths() throws Exception {
        var metrics=new VitMetrics();
        try(var service=new VitService(config("1",4096,4096,4096,0),metrics)) {
            var multi=input("one").toBuilder().addMultimodalInputs(input("two").getMultimodalInputs(0))
                .addMultimodalInputs(input("one").getMultimodalInputs(0)).build();
            VitTest.<MultimodalOutputPB>response(out->service.remoteMultimodalEmbedding(multi,out)).get(3,TimeUnit.SECONDS);
            VitTest.<MultimodalOutputPB>response(out->service.remoteMultimodalEmbedding(multi,out)).get(3,TimeUnit.SECONDS);
            VitTest.<EmptyPB>response(out->service.asyncSubmitEmbedding(input("three"),out)).get(3,TimeUnit.SECONDS);
            // Waiting on the same async media must not create another embedding work item.
            VitTest.<MultimodalOutputPB>response(out->service.remoteMultimodalEmbedding(input("three"),out)).get(3,TimeUnit.SECONDS);
            var text=service.metrics();
            assertEquals(3,sample(text,"py_rtp_vit_qps_total"));
            assertEquals(3,sample(text,"py_rtp_vit_success_qps_total"));
            assertEquals(0,sample(text,"py_rtp_vit_error_qps_total"));
            assertEquals(15,sample(text,"py_rtp_vit_embedding_tps_total"));
            assertEquals(7,sample(text,"py_rtp_vit_image_count_sum"));
            assertEquals(35,sample(text,"py_rtp_vit_embedding_length_sum"));
            assertEquals(3,sample(text,"py_rtp_vit_request_rt_count"));
            assertTrue(text.contains("source=\"mm_embedding\""));
            assertTrue(text.contains("source=\"get_embedding_result\",status=\"success\""));
        }
    }

    VitConfig config(String expression,long device,long host,long hashes,int wait) throws Exception {
        return config(expression,device,host,hashes,wait,Map.of());
    }

    VitConfig config(String expression,long device,long host,long hashes,int wait,Map<String,Object> overrides) throws Exception {
        var path=directory.resolve(UUID.randomUUID()+".json");
        var settings=new HashMap<String,Object>(Map.of("mock.vit.skip.download",true,
            "mm_cache_gpu_max_bytes",device,"mm_cache_cpu_max_bytes",host,"mm_hash_key_cache_max_bytes",hashes,
            "gpu_max_batch_size",8,"gpu_batch_wait_ms",wait,"mock.compute",Map.of("forward",Map.of("expression",expression))));
        settings.putAll(Map.of("patch_size", 14, "downsample_ratio", 3, "min_pixels", 0, "max_image_tokens", 1024, "hidden_size", 4));
        settings.putAll(overrides);
        new ObjectMapper().writeValue(path.toFile(),Map.of("id","vit-0","component_role","vit","component",Map.of("config",settings)));
        return VitConfig.load(path);
    }

    static MultimodalInputsPB input(String key) {
        return MultimodalInputsPB.newBuilder().setRequestId(1).addMultimodalInputs(MultimodalInputPB.newBuilder()
            .setMultimodalType(1).setMultimodalUrl(key).setMmPreprocessConfig(MMPreprocessConfigPB.newBuilder().setWidth(56).setHeight(28))).build();
    }

    static double sample(String text, String name) {
        var values=text.lines().filter(line -> line.startsWith(name + "{") && line.contains("role=\"vit\""))
            .mapToDouble(line -> Double.parseDouble(line.substring(line.lastIndexOf(' ') + 1))).toArray();
        assertTrue(values.length>0,"Missing metric: "+name);
        return Arrays.stream(values).sum();
    }

    @Test void consumersCannotCompleteSharedCacheOrForwardResults() throws Exception {
        var settings=config("100",4096,4096,4096,0);
        var image=new VitGeometry.Image("private-result",56,28,2);
        try(var cache=new VitCache(settings);var scheduler=new VitScheduler(settings)) {
            var claims=cache.acquire(List.of(image.key(),image.key()));
            var first=claims.getFirst(); var second=claims.getLast();
            first.result().toCompletableFuture().complete(image);
            second.result().toCompletableFuture().cancel(true);
            assertFalse(first.result().toCompletableFuture().isDone());
            first.close(); first.close();
            assertTrue(second.hasConsumers());
            cache.publish(first,image,null);
            assertEquals(image,second.result().toCompletableFuture().join());
            second.close(); assertFalse(second.hasConsumers());
            var work=scheduler.submit(image);
            work.result().toCompletableFuture().cancel(true);
            assertEquals(image,work.result().toCompletableFuture().get(3,TimeUnit.SECONDS));
        }
    }

    @Test void sharedWorkFailureIsCountedOnceAndAsyncFailuresAreVisible() throws Exception {
        var metrics=new VitMetrics();
        var settings=config("1",4096,4096,4096,0);
        try(var cache=new VitCache(settings,metrics)) {
            var claims=cache.acquire(List.of("shared","shared"));
            var error=Status.UNAVAILABLE.withDescription("download failed").asRuntimeException();
            metrics.started();
            cache.publish(claims.getFirst(),null,error);
            for(var claim:claims) {
                assertTrue(claim.result().toCompletableFuture().isCompletedExceptionally());
                metrics.error(error,claim);
                metrics.error(Status.DEADLINE_EXCEEDED.asRuntimeException(),claim);
                claim.close();
            }
            String text=org.flexlb.mock.metrics.MetricTags.scrape(metrics.registry,Map.of());
            assertEquals(1,sample(text,"py_rtp_vit_error_qps_total"));
            assertEquals(0,sample(text,"py_rtp_vit_success_qps_total"));
        }
    }

    @Test void gpuEvictionWithoutHostRoomStillCountsLogicalRemoval() throws Exception {
        var metrics=new VitMetrics();
        try(var service=new VitService(config("1",40,0,4096,0),metrics)) {
            for(String key:List.of("first","second"))
                VitTest.<MultimodalOutputPB>response(out->service.remoteMultimodalEmbedding(input(key),out)).get(3,TimeUnit.SECONDS);
            assertEquals(1,sample(service.metrics(),"py_rtp_vit_embedding_cache_eviction_qps_total"));
        }
    }

    @Test void acceptsNativeFrontendTwoHourRpcBudgetForEmbeddingAndHashRequests() throws Exception {
        try (var node=VitNode.start(config("1",1024,1024,1024,0),0,0)) {
            var channel=channel(node);
            try {
                var rpc=MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3,TimeUnit.SECONDS);
                var media=input("native-timeout").toBuilder().setTimeoutMs(7200000).build();
                var output=rpc.remoteMultimodalEmbedding(media);
                assertEquals(List.of(5),output.getSplitSizeList());
                var hashes=rpc.getMultimodalHashes(MultimodalHashRequestPB.newBuilder()
                    .setTimeoutMs(7200000).setInputs(media)
                    .addKeys(MediaInput.key(media.getMultimodalInputs(0))).build());
                assertTrue(hashes.getEntries(0).getHashHit());
            } finally { channel.shutdownNow().awaitTermination(2,TimeUnit.SECONDS); }
        }
    }

    static ManagedChannel channel(VitNode node) {
        return ManagedChannelBuilder.forTarget((String)node.endpoints().get("rpc")).usePlaintext().build();
    }

    @Test void geometryMatchesNativeDsv41PlannerAtPinnedSource() {
        // Values evaluated from ba26ddf1dd plan_image_grid/num_image_tokens, not this Java implementation.
        int[][] vectors = {{1,1,184},{56,28,202},{512,512,184},{1024,1024,652},{1920,1080,968},
            {32768,1,1024},{1,32768,1024},{3840,2160,968},{513,777,268}};
        for (int[] v : vectors) assertEquals(v[2],VitGeometry.rows(v[0],v[1],14,3,295936,1024));
    }

    @Test void statusCountsUniqueMediaUntilPublicationCompletes() throws Exception {
        try (var node=VitNode.start(config("200",1024,1024,1024,0),0,0)) {
            var channel=channel(node);
            try {
                var rpc=MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(2,TimeUnit.SECONDS);
                rpc.asyncSubmitEmbedding(input("shared"));
                rpc.asyncSubmitEmbedding(input("shared"));
                var status=rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance());
                assertTrue(status.hasMmInflightInputs()); assertEquals(1,status.getMmInflightInputs().getValue());
                long until=System.nanoTime()+TimeUnit.SECONDS.toNanos(2);
                while (rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getMmInflightInputs().getValue()!=0) {
                    assertTrue(System.nanoTime()<until); Thread.sleep(5);
                }
            } finally { channel.shutdownNow().awaitTermination(2,TimeUnit.SECONDS); }
        }
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
                    assertEquals(List.of(5L,4L),output.getMultimodalEmbedding().getShapeList());
                    assertTrue(output.getMultimodalEmbedding().getBf16Data().isEmpty());
                    assertEquals(20,output.getMultimodalFeatureHash().getInt32Data().size());assertEquals(List.of(5),output.getSplitSizeList());
                }
                var metadata=MultimodalRpcServiceGrpc.newBlockingStub(channel).getMultimodalHashes(MultimodalHashRequestPB.newBuilder()
                    .addKeys(MediaInput.key(input("image-0").getMultimodalInputs(0))).build());
                assertTrue(metadata.getEntries(0).getHashHit());assertTrue(metadata.getEntries(0).getEmbeddingHit());
                String metrics=java.net.http.HttpClient.newHttpClient().send(java.net.http.HttpRequest.newBuilder(java.net.URI.create((String)node.endpoints().get("metrics"))).GET().build(),java.net.http.HttpResponse.BodyHandlers.ofString()).body();
                assertEquals(1, sample(metrics, "py_rtp_vit_embedding_forward_rt_count"));
                assertEquals(6, sample(metrics, "mock_vit_theoretical_forward_rt_sum"));
                for (String metric : List.of("py_rtp_vit_qps_total", "py_rtp_vit_success_qps_total", "py_rtp_vit_image_count_sum",
                        "py_rtp_vit_embedding_length_sum", "py_rtp_vit_embedding_queue_wait_rt_sum", "py_rtp_vit_embedding_queue_size",
                        "py_rtp_vit_embedding_cache_tokens", "py_rtp_vit_embedding_batch_size_sum"))
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
            assertFalse(joined.producer());assertFalse(joined.result().toCompletableFuture().isDone());
            cache.publish(first,null,new IllegalStateException("failed"));assertTrue(joined.result().toCompletableFuture().isCompletedExceptionally());
            var fresh=cache.acquire(List.of(image.key())).getFirst();cache.publish(first,image,null);assertFalse(fresh.result().toCompletableFuture().isDone());
            cache.publish(fresh,image,null);assertEquals(image,fresh.result().toCompletableFuture().join());
        }
    }

    @Test void oneWaiterDeadlineDoesNotCancelAnotherWaiterOrTheSharedForward() throws Exception {
        try(var node=VitNode.start(config("150",16,16,1024,0),0,0);var threads=Executors.newVirtualThreadPerTaskExecutor()) {
            var channel=channel(node);
            try {
                var shortWait=threads.submit(()->assertThrows(StatusRuntimeException.class,()->MultimodalRpcServiceGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(40,TimeUnit.MILLISECONDS).remoteMultimodalEmbedding(input("same"))));
                var output=MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3,TimeUnit.SECONDS).remoteMultimodalEmbedding(input("same"));
                shortWait.get(3,TimeUnit.SECONDS);assertEquals(5,output.getMultimodalEmbedding().getShape(0));
            } finally { channel.shutdownNow().awaitTermination(2,TimeUnit.SECONDS); }
        }
    }

    @Test void malformedMetadataAndSpanMismatchAreRejectedBeforePAllocation() throws Exception {
        var nativeInput=GenerateInputPB.newBuilder().setRequestId(1).addAllTokenIds(List.of(10,-200,20))
            .addAllMultimodalInputs(input("x").getMultimodalInputsList())
            .build();
        var hashes=VitGeometry.hashes(MediaInput.key(input("x").getMultimodalInputs(0)),2);
        var output=MultimodalOutputPB.newBuilder().addSplitSize(2)
            .setMultimodalEmbedding(TensorPB.newBuilder().setDataType(TensorPB.DataType.BF16).addShape(2).addShape(4))
            .setMultimodalFeatureHash(TensorPB.newBuilder().setDataType(TensorPB.DataType.INT32).addShape(2).setInt32Data(hashes)).build();
        var prepared=MediaInput.apply(nativeInput,output,MediaInput.IMAGE_TOKEN);assertEquals(10,prepared.input().getTokenIds(0));assertEquals(20,prepared.input().getTokenIds(3));
        assertNotEquals(nativeInput.getTokenIdsList(),prepared.input().getTokenIdsList());
        assertThrows(StatusRuntimeException.class,()->MediaInput.apply(nativeInput,output.toBuilder().clearSplitSize().addSplitSize(1).build(),MediaInput.IMAGE_TOKEN));
        assertThrows(StatusRuntimeException.class,()->MediaInput.apply(nativeInput,output.toBuilder().clearMultimodalFeatureHash().build(),MediaInput.IMAGE_TOKEN));
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
            "patch_size", 14, "downsample_ratio", 3, "min_pixels", 0, "max_image_tokens", 1024, "hidden_size", 4,"gpu_max_batch_size",1,"vit_concurrency",1,"vit_max_queue_size",0,"mock.compute",Map.of("forward",Map.of("expression","1000"))))));
        try(var scheduler=new VitScheduler(VitConfig.load(path))) {
            var first=scheduler.submit(new VitGeometry.Image("a",28,28,1));
            long end=System.nanoTime()+TimeUnit.SECONDS.toNanos(2);
            while(scheduler.snapshot().get("mock_vit_running_image_count")==0) { assertTrue(System.nanoTime()<end);Thread.sleep(2); }
            var second=scheduler.submit(new VitGeometry.Image("b",28,28,1));
            assertEquals(Status.Code.RESOURCE_EXHAUSTED,assertThrows(StatusRuntimeException.class,
                ()->scheduler.submit(new VitGeometry.Image("c",28,28,1))).getStatus().getCode());
            scheduler.close();
            assertThrows(ExecutionException.class,()->first.result().toCompletableFuture().get(2,TimeUnit.SECONDS));
            assertThrows(ExecutionException.class,()->second.result().toCompletableFuture().get(2,TimeUnit.SECONDS));
        }
    }

    @Test void embeddingTpsCountsComputedRowsAndExcludesExpiredWork() throws Exception {
        var metrics = new VitMetrics();
        try(var scheduler = new VitScheduler(config("1", 1024, 1024, 1024, 0), metrics)) {
            var expired = scheduler.submit(new VitGeometry.Image("expired", 28, 28, 100), System.nanoTime() - 1);
            assertThrows(ExecutionException.class, () -> expired.result().toCompletableFuture().get(2, TimeUnit.SECONDS));
            var first = scheduler.submit(new VitGeometry.Image("first", 28, 28, 10));
            var second = scheduler.submit(new VitGeometry.Image("second", 28, 28, 22));
            first.result().toCompletableFuture().get(2, TimeUnit.SECONDS); second.result().toCompletableFuture().get(2, TimeUnit.SECONDS);
            assertEquals(32, sample(org.flexlb.mock.metrics.MetricTags.scrape(metrics.registry, Map.of()), "py_rtp_vit_embedding_tps_total"));
            assertEquals(32, sample(org.flexlb.mock.metrics.MetricTags.scrape(metrics.registry, Map.of()), "py_rtp_vit_embedding_tps_total"));
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
                assertEquals(0, sample(metrics, "py_rtp_vit_embedding_cache_eviction_qps_total"));
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
            first.result().toCompletableFuture().get(2,TimeUnit.SECONDS);
            assertThrows(ExecutionException.class,()->abandoned.result().toCompletableFuture().get(2,TimeUnit.SECONDS));
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
            notified.toCompletableFuture().get(2,TimeUnit.SECONDS); claims.forEach(VitCache.Claim::close);
            long until=System.nanoTime()+TimeUnit.SECONDS.toNanos(2);
            while(preparation.inflight()!=0) { assertTrue(System.nanoTime()<until);Thread.sleep(2); }
            var next=preparation.submit(input("second"));
            next.getFirst().result().toCompletableFuture().get(2,TimeUnit.SECONDS);next.forEach(VitCache.Claim::close);
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

    @Test void healthDoesNotWaitForVisionSchedulingLock() throws Exception {
        try (var scheduler = new VitScheduler(config("1",1024,1024,1024,0));
             var readers = Executors.newVirtualThreadPerTaskExecutor()) {
            synchronized (scheduler) {
                readers.submit(scheduler::checkHealthy).get(1, TimeUnit.SECONDS);
            }
            scheduler.close();
            assertThrows(IllegalStateException.class, scheduler::checkHealthy);
        }
    }

    private static <T> CompletableFuture<T> response(java.util.function.Consumer<io.grpc.stub.StreamObserver<T>> call) {
        var result=new CompletableFuture<T>();
        call.accept(new io.grpc.stub.StreamObserver<>() {
            public void onNext(T value) { result.complete(value); }
            public void onError(Throwable error) { result.completeExceptionally(error); }
            public void onCompleted() {}
        });
        return result;
    }

}
