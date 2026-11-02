package org.flexlb.mock.vit;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import io.grpc.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;
import javax.imageio.ImageIO;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class VitDownloadTest {
    @TempDir Path directory;
    private VitConfig config(int concurrency,int waiting) throws Exception {
        var json=new ObjectMapper();
        json.writeValue(directory.resolve("config.json").toFile(),Map.of("image_token_id",1000,
            "text_config",Map.of("hidden_size",4),"vision_config",Map.of("patch_size",14,"downsample_ratio",3,"min_pixels",0,"max_image_tokens",1024)));
        return VitConfig.load(json.valueToTree(Map.of("id","v","component_role","vit","component",Map.of("config",Map.of(
            "checkpoint_path",directory.toString(),"vit_concurrency",concurrency,"vit_max_queue_size",waiting,
            "gpu_max_batch_size",4,"gpu_batch_wait_ms",0,"mm_image_max_file_size_kb",1,
            "download_headers","{\"X-Media-Test\":\"present\"}","mock.compute.forward.expression","5")))));
    }
    private static byte[] png() throws Exception {
        var bytes=new ByteArrayOutputStream(); ImageIO.write(new BufferedImage(56,28,BufferedImage.TYPE_INT_RGB),"png",bytes); return bytes.toByteArray();
    }
    private static MultimodalInputsPB input(HttpServer server,String path) {
        // Deliberately wrong hints: download mode must inspect the actual image.
        return VitTest.input("http://127.0.0.1:"+server.getAddress().getPort()+path).toBuilder().setRequestId(1)
            .setMultimodalInputs(0,VitTest.input("http://127.0.0.1:"+server.getAddress().getPort()+path).getMultimodalInputs(0).toBuilder()
                .setMmPreprocessConfig(MMPreprocessConfigPB.newBuilder().setWidth(1).setHeight(1))).build();
    }
    private static void await(BooleanSupplier condition) throws Exception {
        long end=System.nanoTime()+TimeUnit.SECONDS.toNanos(5);
        while(!condition.getAsBoolean()) { assertTrue(System.nanoTime()<end,"Timed out waiting for V state"); Thread.sleep(5); }
    }

    private static String metrics(VitNode node) throws Exception {
        try(var client=java.net.http.HttpClient.newHttpClient()) {
            return client.send(java.net.http.HttpRequest.newBuilder(URI.create((String)node.endpoints().get("metrics"))).GET().build(),
                java.net.http.HttpResponse.BodyHandlers.ofString()).body();
        }
    }

    @Test void blockedDownloadDoesNotBlockForwardAndNativeAdmissionCountsUniqueMedia() throws Exception {
        var firstEntered=new CountDownLatch(1);var secondEntered=new CountDownLatch(1);
        var releaseFirst=new CountDownLatch(1);var releaseSecond=new CountDownLatch(1);
        var hits=new ConcurrentHashMap<String,AtomicInteger>(); byte[] image=png();
        var server=HttpServer.create(new InetSocketAddress("127.0.0.1",0),0);
        try(var threads=Executors.newVirtualThreadPerTaskExecutor()) {
            server.setExecutor(threads);
            server.createContext("/",exchange->{
                String path=exchange.getRequestURI().getPath(); hits.computeIfAbsent(path,ignored->new AtomicInteger()).incrementAndGet();
                try {
                    exchange.sendResponseHeaders(200,image.length);
                    if(path.equals("/first")) { firstEntered.countDown();releaseFirst.await(); }
                    if(path.equals("/second")) { secondEntered.countDown();releaseSecond.await(); }
                    exchange.getResponseBody().write(image);
                } catch(InterruptedException error) { Thread.currentThread().interrupt(); }
                finally { exchange.close(); }
            });server.start();
            try(var node=VitNode.start(config(2,1),0,0)) {
                var channel=VitTest.channel(node);
                try {
                    var rpc=MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(5,TimeUnit.SECONDS);
                    rpc.asyncSubmitEmbedding(input(server,"/first"));assertTrue(firstEntered.await(2,TimeUnit.SECONDS));
                    rpc.asyncSubmitEmbedding(input(server,"/first")); // Same key shares its producer/slot.
                    rpc.asyncSubmitEmbedding(input(server,"/second"));assertTrue(secondEntered.await(2,TimeUnit.SECONDS));
                    rpc.asyncSubmitEmbedding(input(server,"/queued"));
                    assertEquals(3,rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getMmInflightInputs().getValue());
                    assertFalse(hits.containsKey("/queued"));
                    String snapshot=metrics(node);
                    assertEquals(3,VitTest.sample(snapshot,"py_rtp_vit_preprocess_queue_size"));
                    assertEquals(0,VitTest.sample(snapshot,"py_rtp_vit_embedding_queue_size"));
                    assertEquals(Status.Code.RESOURCE_EXHAUSTED,assertThrows(StatusRuntimeException.class,
                        ()->rpc.asyncSubmitEmbedding(input(server,"/overflow"))).getStatus().getCode());
                    releaseSecond.countDown();
                    var output=rpc.remoteMultimodalEmbedding(input(server,"/second"));
                    assertEquals(List.of(5L,4L),output.getMultimodalEmbedding().getShapeList());
                    rpc.remoteMultimodalEmbedding(input(server,"/queued"));
                    assertEquals(1,releaseFirst.getCount()); // Forward completed while first HTTP body is still blocked.
                    assertEquals(1,hits.get("/first").get());
                    releaseFirst.countDown();
                    await(()->rpc.withDeadlineAfter(1,TimeUnit.SECONDS).getWorkerStatus(StatusVersionPB.getDefaultInstance()).getMmInflightInputs().getValue()==0);
                } finally { releaseFirst.countDown();releaseSecond.countDown();channel.shutdownNow().awaitTermination(2,TimeUnit.SECONDS); }
            } finally { server.stop(0); }
        }
    }

    @Test void queuedCancellationFreesAdmissionWithoutDownloadingAndStopCancelsTheActiveRead() throws Exception {
        var entered=new CountDownLatch(1);var release=new CountDownLatch(1);var queuedHits=new AtomicInteger();byte[] image=png();
        var server=HttpServer.create(new InetSocketAddress("127.0.0.1",0),0);
        try(var threads=Executors.newVirtualThreadPerTaskExecutor()) {
            server.setExecutor(threads);server.createContext("/",exchange->{
                try {
                    exchange.sendResponseHeaders(200,image.length);
                    if(exchange.getRequestURI().getPath().equals("/slow")) { entered.countDown();release.await(); }
                    else queuedHits.incrementAndGet();
                    exchange.getResponseBody().write(image);
                } catch(InterruptedException error) { Thread.currentThread().interrupt(); }
                finally { exchange.close(); }
            });server.start();
            var node=VitNode.start(config(1,1),0,0);var channel=VitTest.channel(node);
            try {
                var rpc=MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3,TimeUnit.SECONDS);
                rpc.asyncSubmitEmbedding(input(server,"/slow"));assertTrue(entered.await(2,TimeUnit.SECONDS));
                assertEquals(Status.Code.DEADLINE_EXCEEDED,assertThrows(StatusRuntimeException.class,()->
                    rpc.withDeadlineAfter(100,TimeUnit.MILLISECONDS).remoteMultimodalEmbedding(input(server,"/queued"))).getStatus().getCode());
                await(()->rpc.getWorkerStatus(StatusVersionPB.getDefaultInstance()).getMmInflightInputs().getValue()==1);
                assertEquals(0,queuedHits.get());
                threads.submit(node::close).get(3,TimeUnit.SECONDS);
            } finally { release.countDown();node.close();channel.shutdownNow().awaitTermination(2,TimeUnit.SECONDS);server.stop(0); }
        }
    }

    @Test void failedReadsAndBodyDeadlinesReleaseAdmissionWithoutUsingWidthHints() throws Exception {
        byte[] image=png();var stalled=new CountDownLatch(1);
        var server=HttpServer.create(new InetSocketAddress("127.0.0.1",0),0);
        try(var threads=Executors.newVirtualThreadPerTaskExecutor()) {
            server.setExecutor(threads);server.createContext("/",exchange->{
                String path=exchange.getRequestURI().getPath();
                byte[] bytes=path.equals("/large")?new byte[2048]:path.equals("/bad")?new byte[]{1,2,3}:image;
                try {
                    int status=path.equals("/missing")?404:"present".equals(exchange.getRequestHeaders().getFirst("X-Media-Test"))?200:403;
                    exchange.sendResponseHeaders(status,bytes.length);
                    if(path.equals("/timeout")) stalled.await();
                    exchange.getResponseBody().write(bytes);
                } catch(InterruptedException error) { Thread.currentThread().interrupt(); }
                finally { exchange.close(); }
            });server.start();
            try(var node=VitNode.start(config(1,0),0,0)) {
                var channel=VitTest.channel(node);
                try {
                    var rpc=MultimodalRpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(3,TimeUnit.SECONDS);
                    for(String path:List.of("/missing","/large","/bad","/timeout")) {
                        var request=input(server,path).toBuilder().setTimeoutMs(150).build();
                        var error=assertTimeoutPreemptively(java.time.Duration.ofSeconds(2),
                            ()->assertThrows(StatusRuntimeException.class,()->rpc.withDeadlineAfter(3,TimeUnit.SECONDS).remoteMultimodalEmbedding(request)));
                        var expected=path.equals("/missing")?Status.Code.UNAVAILABLE:path.equals("/timeout")?Status.Code.DEADLINE_EXCEEDED:Status.Code.INVALID_ARGUMENT;
                        assertEquals(expected,error.getStatus().getCode());
                        await(()->rpc.withDeadlineAfter(1,TimeUnit.SECONDS).getWorkerStatus(StatusVersionPB.getDefaultInstance()).getMmInflightInputs().getValue()==0);
                    }
                    var urlOnly=input(server,"/good").toBuilder();
                    urlOnly.setMultimodalInputs(0,urlOnly.getMultimodalInputs(0).toBuilder().clearMmPreprocessConfig());
                    assertEquals(5,rpc.withDeadlineAfter(3,TimeUnit.SECONDS).remoteMultimodalEmbedding(urlOnly.build()).getMultimodalEmbedding().getShape(0));
                    rpc.withDeadlineAfter(3,TimeUnit.SECONDS).remoteMultimodalEmbedding(urlOnly.build()); // Cache hit is still a request, not a Forward.
                    String snapshot=metrics(node);
                    assertEquals(6,VitTest.sample(snapshot,"py_rtp_vit_request_rt_count"));
                    assertEquals(1,VitTest.sample(snapshot,"py_rtp_vit_embedding_rt_count"));
                    assertEquals(5,VitTest.sample(snapshot,"py_rtp_vit_download_rt_count"));
                    assertEquals(0,VitTest.sample(snapshot,"py_rtp_vit_preprocess_queue_size"));
                } finally { channel.shutdownNow().awaitTermination(2,TimeUnit.SECONDS); }
            } finally { stalled.countDown();server.stop(0); }
        }
    }
}
