package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.*;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

/** Actual dry-run sends must match offline reconstruction across laps and uneven shards. */
class PlaybackPlanTest {
    @TempDir Path temp;
    static final ObjectMapper JSON=new ObjectMapper();
    private JavaLoadClient.Config config(Path trace,Path out,int shards,int shard,String mode) {
        return new JavaLoadClient.Config(trace.toString(),"127.0.0.1:7001","127.0.0.1:7003",
            0,16,1.0,1,out.toString(),shards,shard,0,120000L,500.0,false,true,1,1,0L,120,true,
            "engine_service","",0,0,"",false,"",true,0,0,mode,100,0,false,List.of());
    }

    private List<JsonNode> read(Path path) throws Exception {
        List<JsonNode> out=new ArrayList<>();
        for(String line:Files.readAllLines(path)) out.add(JSON.readTree(line));
        return out;
    }
    @Test void dryRunSnapshotReconstructsMultiLapIdentitiesShardsAndReplay() throws Exception {
        Path trace=temp.resolve("trace.jsonl");
        StringBuilder source=new StringBuilder();
        for(int i=0;i<5;i++) source.append("{\"rid\":\"r").append(i)
            .append("\",\"ts\":").append(i*10).append(",\"il\":1,\"ol\":4,\"priority\":50,\"cache_key_block_size\":512,\"input_token_blocks\":[1]}\n");
        Files.writeString(trace,source);
        for(String mode:List.of("uniform","replay")) for(String identity:List.of("none","structural-relabel")) {
            for(int shard=0;shard<3;shard++) {
                Path out=temp.resolve(mode+identity+shard);
                Playback policy=new Playback(Map.of("MAX_LAPS","3","LAP_IDENTITY",identity),false,0);
                try(JavaLoadClient client=new JavaLoadClient(config(trace,out,3,shard,mode))) { client.run(policy); }
                Path plan=out.resolve("plan.jsonl");
                PlaybackPlan.write(out.resolve("playback.json"),trace,plan);
                List<JsonNode> expected=read(plan),actual=read(out.resolve("client_events.jsonl"));
                assertEquals(shard==2?3:6,expected.size()); assertEquals(expected.size(),actual.size());
                actual.sort(Comparator.comparingDouble(r->r.path("send_due_epoch_ms").asDouble()));
                double origin=actual.get(0).path("send_due_epoch_ms").asDouble()-expected.get(0).path("due_seconds").asDouble()*1000;
                for(int i=0;i<expected.size();i++) {
                    JsonNode e=expected.get(i),a=actual.get(i);
                    assertEquals(e.path("rid").asText(),a.path("rid").asText());
                    assertEquals(e.path("ol").asInt(),a.path("output_len").asInt());
                    assertEquals(e.path("due_seconds").asDouble(),(a.path("send_due_epoch_ms").asDouble()-origin)/1000,0.0011);
                    // The input token stride is 2. Fresh laps must have fresh actual block keys.
                    var tokens=List.of(identity.equals("none")?1:1+e.path("iteration").asInt()*2);
                    assertEquals(JSON.valueToTree(JavaLoadClient.computeBlockKeys(tokens,512)),e.path("block_keys"));
                }
                Path again=out.resolve("again.jsonl");PlaybackPlan.write(out.resolve("playback.json"),trace,again);
                assertEquals(Files.readString(plan),Files.readString(again));
                Path wrong=temp.resolve("wrong");Files.writeString(wrong,source+"\n");
                assertThrows(IllegalArgumentException.class,()->PlaybackPlan.write(out.resolve("playback.json"),wrong,again));
            }
        }
    }
}
