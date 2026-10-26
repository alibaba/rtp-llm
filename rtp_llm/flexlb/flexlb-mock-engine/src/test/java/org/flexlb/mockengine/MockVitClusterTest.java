package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.*;
import io.grpc.*;
import org.flexlb.engine.grpc.*;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;
import java.net.*;
import java.net.http.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import static org.junit.jupiter.api.Assertions.*;

/** Real sockets to one cluster JVM: independent V workers and common lifecycle/discovery. */
class MockVitClusterTest {
    @TempDir Path directory;
    final ObjectMapper json=new ObjectMapper();
    final HttpClient http=HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(1)).build();
    Process child; int base; final List<ManagedChannel> channels=new ArrayList<>();
    @BeforeEach void start() throws Exception {
        while(true) {
            base=java.util.concurrent.ThreadLocalRandom.current().nextInt(20000,27000);
            var held=new ArrayList<ServerSocket>();
            try { for(int i=-1;i<8;i++)held.add(new ServerSocket(base+i)); break; }
            catch(java.io.IOException occupied) { }
            finally { for(var socket:held)socket.close(); }
        }
        Path calibration=directory.resolve("performance.json");
        Files.writeString(calibration,"{\"mock\":{\"forward\":{\"prefill\":{\"fixed_ms\":1},\"vit\":{\"fixed_ms\":300}},\"vit\":{\"hidden_size\":16,\"max_output_bytes\":16777216}}}");
        child=new ProcessBuilder(System.getProperty("java.home")+"/bin/java","-cp",System.getProperty("java.class.path"),
                JavaMockEngineCluster.class.getName(),"--mock.engine_config","../tools/mock_bundle/engine-config.json",
                "--mock.calibration",calibration.toString(),"--mock.n_prefill","1","--mock.n_decode","1","--mock.n_vit","2",
                "--mock.unique_engine_ips","false","--mock.base_grpc_port",Integer.toString(base),
                "--mock.endpoint_file",directory.resolve("endpoints.json").toString())
                .redirectErrorStream(true).redirectOutput(directory.resolve("engine.log").toFile()).start();
        await(()->{
            if(!child.isAlive())fail(Files.readString(directory.resolve("engine.log")));
            try{return get("/health").path("healthy").asBoolean();}catch(java.io.IOException starting){return false;}
        });
    }
    @AfterEach void close() throws Exception {
        for(var channel:channels)channel.shutdownNow().awaitTermination(2,TimeUnit.SECONDS);
        if(child!=null){child.destroy();if(!child.waitFor(10,TimeUnit.SECONDS))child.destroyForcibly().waitFor();}
    }
    interface Check { boolean ok() throws Exception; }
    void await(Check check) throws Exception {
        long until=System.nanoTime()+TimeUnit.SECONDS.toNanos(8);
        while(System.nanoTime()<until){if(check.ok())return;Thread.sleep(10);}assertTrue(check.ok());
    }
    ManagedChannel channel(int port) {var c=ManagedChannelBuilder.forAddress("127.0.0.1",port).usePlaintext().build();channels.add(c);return c;}
    MultimodalRpcServiceGrpc.MultimodalRpcServiceBlockingStub vit(int port) {return MultimodalRpcServiceGrpc.newBlockingStub(channel(port)).withDeadlineAfter(4,TimeUnit.SECONDS);}
    HttpResponse<String> post(String path,String body) throws Exception {
        return http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:"+(base-1)+path)).timeout(Duration.ofSeconds(8))
                .POST(HttpRequest.BodyPublishers.ofString(body)).build(),HttpResponse.BodyHandlers.ofString());
    }
    JsonNode get(String path) throws Exception {
        var response=http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:"+(base-1)+path)).timeout(Duration.ofSeconds(1)).GET().build(),HttpResponse.BodyHandlers.ofString());
        assertEquals(200,response.statusCode());return json.readTree(response.body());
    }
    JsonNode snap(int port) throws Exception {for(var node:get("/snapshot").path("engines"))if(node.path("grpc_addr").asText().endsWith(":"+port))return node;throw new AssertionError("missing worker");}
    JsonNode discovery() throws Exception {return json.readTree(directory.resolve("discovery.json").toFile());}
    void assertPdAlive() {for(int port:new int[]{base,base+1})assertTrue(RpcServiceGrpc.newBlockingStub(channel(port)).withDeadlineAfter(2,TimeUnit.SECONDS).getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance()).getAlive());}

    @Test void sharedProcessHasIndependentCachesAndOneAtomicDiscoveryOwner() throws Exception {
        assertEquals(4,get("/snapshot").path("engines").size());
        assertEquals(0,child.descendants().count(),"all four workers are hosted by this JVM");
        assertEquals(2,discovery().path("mock.vit.hosts.address").size());
        var image=MockVitServiceTest.image(56,56,0x123456);var input=MockVitServiceTest.inputs(image);String key=MockVitImage.key(image);
        var a=vit(base+2);var b=vit(base+3);
        a.remoteMultimodalEmbedding(input);a.remoteMultimodalEmbedding(input);
        var probe=EngineRpcService.MultimodalHashRequestPB.newBuilder().addKeys(key).build();
        assertTrue(a.getMultimodalHashes(probe).getEntries(0).getEmbeddingHit());
        assertFalse(b.getMultimodalHashes(probe).getEntries(0).getHashHit());
        var identity=a.getMultimodalHashes(probe).getWorkerInstance();
        assertNotEquals(identity,b.getMultimodalHashes(probe).getWorkerInstance());
        assertEquals(1,snap(base+2).path("vit").path("computations").asInt());
        assertEquals(0,snap(base+3).path("vit").path("computations").asInt());
        var added=post("/add_engine","{\"role\":\"vit\"}");assertEquals(200,added.statusCode(),added.body());
        int port=json.readTree(added.body()).path("port").asInt();
        assertEquals(3,discovery().path("mock.vit.hosts.address").size());
        assertFalse(vit(port).getMultimodalHashes(probe).getEntries(0).getHashHit());
        var removed=post("/remove_engine","{\"port\":"+(base+2)+"}");assertEquals(200,removed.statusCode(),removed.body());
        assertTrue(json.readTree(removed.body()).path("drained").asBoolean());
        assertEquals(2,discovery().path("mock.vit.hosts.address").size());
        assertEquals(200,post("/add_engine","{\"role\":\"vit\",\"port\":"+(base+2)+"}").statusCode());
        assertNotEquals(identity,vit(base+2).getMultimodalHashes(probe).getWorkerInstance());
        assertPdAlive();
    }
    @Test void vStopRestartAndFailedBindDoNotAffectPdOrOtherV() throws Exception {
        var v=vit(base+2);var status=EngineRpcService.StatusVersionPB.getDefaultInstance();
        assertTrue(v.getWorkerStatus(status).getAlive());
        assertEquals(200,post("/stop_engine","{\"engine\":\"vit-0\"}").statusCode());
        assertThrows(StatusRuntimeException.class,()->v.getWorkerStatus(status));
        assertPdAlive();assertTrue(vit(base+3).getWorkerStatus(status).getAlive());
        assertEquals(200,post("/start_engine","{\"engine\":\"vit-0\"}").statusCode());
        assertTrue(vit(base+2).getWorkerStatus(status).getAlive());
        try(var occupied=new ServerSocket(0)) {
            var response=post("/add_engine","{\"role\":\"vit\",\"port\":"+occupied.getLocalPort()+"}");
            assertEquals(500,response.statusCode());
            assertEquals(4,get("/snapshot").path("engines").size());assertEquals(2,discovery().path("mock.vit.hosts.address").size());
        }
        assertPdAlive();
    }
    @Test void gracefulVRemovalWithdrawsBeforeDrainAndCompletesAcceptedImage() throws Exception {
        var stub=MultimodalRpcServiceGrpc.newFutureStub(channel(base+2)).withDeadlineAfter(5,TimeUnit.SECONDS);
        var future=stub.remoteMultimodalEmbedding(MockVitServiceTest.inputs(MockVitServiceTest.image(56,56,0x987654)));
        await(()->snap(base+2).path("vit").path("active").asInt()>0);
        var remove=CompletableFuture.supplyAsync(()->{try{return post("/remove_engine","{\"engine\":\"vit-0\",\"drain_timeout_ms\":5000}");}catch(Exception e){throw new CompletionException(e);}});
        await(()->discovery().path("mock.vit.hosts.address").size()==1);
        assertFalse(remove.isDone());assertPdAlive();
        var error=assertThrows(StatusRuntimeException.class,()->vit(base+2).remoteMultimodalEmbedding(MockVitServiceTest.inputs(MockVitServiceTest.image(56,56,0x112233))));
        assertEquals(Status.Code.UNAVAILABLE,error.getStatus().getCode());
        assertTrue(future.get(4,TimeUnit.SECONDS).hasMultimodalEmbedding());
        var result=remove.get(6,TimeUnit.SECONDS);assertEquals(200,result.statusCode(),result.body());
        assertTrue(json.readTree(result.body()).path("drained").asBoolean());
        assertEquals(3,get("/snapshot").path("engines").size());
    }
    @Test void vUsesCommonTimingControlAndPublishesItsOwnMetrics() throws Exception {
        var response=post("/set_perf","{\"engine\":\"vit-0\",\"mock\":{\"timing\":{\"seed\":42,\"time_scale\":0.01},\"forward\":{\"vit\":{\"fixed_ms\":1000,\"pause\":{\"probability\":1,\"duration_ms\":1000}}}}}");
        assertEquals(200,response.statusCode(),response.body());
        var stub=MultimodalRpcServiceGrpc.newBlockingStub(channel(base+2)).withDeadlineAfter(1500,TimeUnit.MILLISECONDS);
        assertTrue(stub.remoteMultimodalEmbedding(MockVitServiceTest.inputs(MockVitServiceTest.image(56,56,0xabcdef))).hasMultimodalEmbedding());
        boolean observed=false;
        for(var engine:get("/effective_config").path("engines"))if(engine.path("engine").asText().equals("vit-0")) {
            assertEquals(1,engine.path("calibration").path("generation").asInt());
            assertEquals(0.01,engine.path("calibration").path("mock").path("timing").path("time_scale").asDouble()); observed=true;
        }
        assertTrue(observed);
        var metrics=http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:"+(base-1)+"/metrics?per_engine=true")).GET().build(),HttpResponse.BodyHandlers.ofString());
        assertTrue(metrics.body().contains("mock_vit_computations_total{engine_name=\"vit-0\""));
        assertTrue(metrics.body().contains("mock_vit_computations_total{engine_name=\"vit-1\""));
        assertPdAlive();
    }

    @Test void prefillFetchesFeaturesFromSelectedVWithinTheSameClusterJvm() throws Exception {
        var image=MockVitServiceTest.image(56,56,0xaabbcc);
        var features=vit(base+2).remoteMultimodalEmbedding(MockVitServiceTest.inputs(image));
        var hashes=features.getMultimodalFeatureHash().getInt32Data().asReadOnlyByteBuffer().order(java.nio.ByteOrder.LITTLE_ENDIAN);
        int rows=hashes.remaining()/4;
        var request=EngineRpcService.GenerateInputPB.newBuilder().setRequestId(101).addTokenIds(10);
        while(hashes.hasRemaining())request.addTokenIds(hashes.getInt());
        request.addTokenIds(11).addMultimodalInputs(image);
        request.getMultimodalTokenLayoutBuilder().addSpansBuilder().setOffset(1).setLength(rows);
        request.getGenerateConfigBuilder().setMaxNewTokens(2).setTimeoutMs(5000)
                .addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder().setRoleStr("VIT").setRole(EngineRpcService.RoleAddrPB.RoleType.VIT).setIp("127.0.0.1").setGrpcPort(base+2))
                .addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder().setRoleStr("DECODE").setRole(EngineRpcService.RoleAddrPB.RoleType.DECODE).setIp("127.0.0.1").setGrpcPort(base+1));
        var stream=RpcServiceGrpc.newBlockingStub(channel(base)).withDeadlineAfter(7,TimeUnit.SECONDS).generateStreamCall(request.build());
        boolean finished=false;
        while(stream.hasNext()) {
            var frame=stream.next();assertFalse(frame.hasErrorInfo(),frame.toString());
            if(frame.hasFlattenOutput() && frame.getFlattenOutput().getFinishedCount()>0)finished|=frame.getFlattenOutput().getFinished(0);
        }
        assertTrue(finished);
        await(()->snap(base).path("running").asInt()==0 && snap(base+1).path("running").asInt()==0);
        assertEquals(1,snap(base+2).path("vit").path("computations").asInt());
        assertEquals(0,snap(base+3).path("vit").path("computations").asInt());
    }

    @Test void slowVForwardDoesNotOccupyAnotherWorkersComputeLane() throws Exception {
        assertEquals(200,post("/set_perf","{\"engine\":\"vit-0\",\"mock\":{\"forward\":{\"vit\":{\"fixed_ms\":4000}}}}").statusCode());
        assertEquals(200,post("/set_perf","{\"engine\":\"vit-1\",\"mock\":{\"forward\":{\"vit\":{\"fixed_ms\":0}}}}").statusCode());
        var slow=MultimodalRpcServiceGrpc.newFutureStub(channel(base+2)).withDeadlineAfter(6,TimeUnit.SECONDS)
                .remoteMultimodalEmbedding(MockVitServiceTest.inputs(MockVitServiceTest.image(56,56,0x112233)));
        try {
            await(()->snap(base+2).path("vit").path("active").asInt()>0);
            var fast=MultimodalRpcServiceGrpc.newBlockingStub(channel(base+3)).withDeadlineAfter(2,TimeUnit.SECONDS);
            assertTrue(fast.remoteMultimodalEmbedding(MockVitServiceTest.inputs(MockVitServiceTest.image(56,56,0x112233))).hasMultimodalEmbedding());
            assertFalse(slow.isDone(),"V compute lanes must remain independent within the JVM");
            assertPdAlive();
        } finally { slow.cancel(true); }
    }

}
