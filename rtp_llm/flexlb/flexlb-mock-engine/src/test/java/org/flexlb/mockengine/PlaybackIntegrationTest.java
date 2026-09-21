package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.*;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;
import static org.mockito.ArgumentMatchers.any;

/** Real sender loop and per-request evidence; Schedule is stubbed, no GPU claim. */
class PlaybackIntegrationTest {
    @TempDir Path root;
    private static final ObjectMapper JSON = new ObjectMapper();

    List<com.fasterxml.jackson.databind.JsonNode> send(String mode, long[] times, Map<String,String> policy) throws Exception {
        Path trace=root.resolve("trace-"+mode+".jsonl"),out=root.resolve("out-"+mode);
        StringBuilder rows=new StringBuilder();
        for(int i=0;i<times.length;i++) rows.append("{\"rid\":\"r"+i+"\",\"ts\":"+times[i]+",\"il\":512,\"ol\":1,\"priority\":50,\"cache_key_block_size\":512,\"input_token_blocks\":[1]}\n");
        Files.writeString(trace,rows);
        JavaLoadClient.Config config=new JavaLoadClient.Config(
            trace.toString(),"127.0.0.1:1","127.0.0.1:1",2,64,4.0,1,out.toString(),1,0,0,
            1000L,500.0,false,false,1,1,0L,2,true,"engine_service","",
            false,10,1000,0,0,"",false,"",false,0,0,mode,240.0,false);
        try(JavaLoadClient client=new JavaLoadClient(config)) {
            var field=JavaLoadClient.class.getDeclaredField("scheduleStubs");field.setAccessible(true);
            var stubs=(FlexlbServiceGrpc.FlexlbServiceBlockingStub[])field.get(client);
            var stub=mock(FlexlbServiceGrpc.FlexlbServiceBlockingStub.class,withSettings().defaultAnswer(RETURNS_SELF));
            when(stub.schedule(any())).thenReturn(FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                .setCode(200).setSuccess(true).setEnqueuedByMaster(true).build());
            stubs[0]=stub;
            client.run(new Playback(policy,false,2));
        }
        List<com.fasterxml.jackson.databind.JsonNode> result=new ArrayList<>();
        for(String line:Files.readAllLines(out.resolve("client_events.jsonl"))) result.add(JSON.readTree(line));
        result.sort(Comparator.comparingDouble(r->r.path("send_due_epoch_ms").asDouble()));
        return result;
    }
    @Test void finiteUniform240IgnoresTraceTimesWithoutImplicitWrap() throws Exception {
        var rows=send("uniform",new long[]{0,10000,40000,90000},Map.of());
        assertEquals(4,rows.size());
        double first=rows.get(0).path("send_due_epoch_ms").asDouble();
        for(int i=0;i<rows.size();i++) {
            assertEquals(i*1000.0/240,rows.get(i).path("send_due_epoch_ms").asDouble()-first,0.001);
            assertEquals(0,rows.get(i).path("iteration").asInt(-1));
        }
    }
    @Test void replayUsesTrueTimestampsDividedByFourAndWallDuration() throws Exception {
        var rows=send("replay",new long[]{0,120,400,1200,6000,20000},Map.of());
        assertEquals(5,rows.size()); // 6000ms source timestamp remains within 2s at 4x.
        double first=rows.get(0).path("send_due_epoch_ms").asDouble();
        double[] expected={0,30,100,300,1500};
        for(int i=0;i<rows.size();i++) {
            assertEquals(expected[i],rows.get(i).path("send_due_epoch_ms").asDouble()-first,0.001);
            assertTrue(rows.get(i).path("pacing_lag_ms").asDouble()<250);
        }
    }
    @Test void explicitLapsAreReportedWithDisjointRequestIds() throws Exception {
        var rows=send("uniform",new long[]{0,100},Map.of("MAX_LAPS","2"));
        assertEquals(4,rows.size());
        assertEquals(List.of(0,0,1,1),rows.stream().map(r->r.path("iteration").asInt(-1)).toList());
        assertEquals(4,rows.stream().map(r->r.path("rid").asText()).distinct().count());
    }
}
