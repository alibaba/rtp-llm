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
    @Test void manualAliasesDefaultsAndSkippedZeroOutputsMatchActualSends() throws Exception {
        Path trace = temp.resolve("aliases.jsonl"), out = temp.resolve("aliases-out");
        Files.writeString(trace, "{\"rid\":\"skipped\",\"ts\":0,\"il\":1,\"ol\":0,\"input_ids\":[1]}\n"
                + "{\"request_id\":\"accepted\",\"ts_epoch_ms\":10,\"input_token_len\":1,\"output_token_len\":2,\"input_ids\":[1],\"bh\":[99]}\n");
        try (var client = new JavaLoadClient(config(trace, out, 1, 0, "replay"))) {
            client.run(new Playback(Map.of("MAX_LAPS", "1"), false, 0));
        }
        Path plan = out.resolve("plan.jsonl");
        PlaybackPlan.write(out.resolve("playback.json"), trace, plan);
        var rows = read(plan);
        assertEquals(1, rows.size());
        assertEquals("accepted", rows.get(0).path("rid").asText());
        assertEquals(JSON.readTree("[99]"), rows.get(0).path("block_keys"));
    }

    @Test void invalidUnboundedOfflineReplayPreservesExistingOutput() throws Exception {
        Path trace = temp.resolve("bounded.jsonl"), snapshot = temp.resolve("snapshot.json"), output = temp.resolve("frozen-plan.jsonl");
        Files.writeString(trace, "{\"rid\":\"a\",\"ts\":0,\"il\":1,\"ol\":2,\"input_ids\":[1]}\n");
        var policy = JSON.createObjectNode();
        policy.put("planner_version", "PLAYBACK_V1").put("offline_plan_supported", true)
                .put("trace_sha256", PlaybackPlan.sha256(trace)).put("num_shards", 1)
                .put("shard_index", 0).put("duration_seconds", 1).put("max_laps", 0)
                .put("identity", "none").put("mode", "replay").put("speed", 0).put("token_stride", 2);
        JSON.writeValue(snapshot.toFile(), policy);
        Files.writeString(output, "previous complete output");
        assertThrows(IllegalArgumentException.class, () -> PlaybackPlan.write(snapshot, trace, output));
        assertEquals("previous complete output", Files.readString(output));
    }

    @Test void skippedMalformedTokensCannotPoisonStrideAndHashOnlyLapsKeepTheirPrefixes() throws Exception {
        try (var client = new JavaLoadClient(config(temp.resolve("unused"), temp.resolve("unused-out"), 1, 0, "replay"))) {
            assertThrows(IllegalArgumentException.class, () -> client.parseTraceRecord(JSON.readTree(
                    "{\"rid\":\"bad\",\"il\":1,\"ol\":2,\"input_ids\":[2147483648]}")));
            client.parseTraceRecord(JSON.readTree("{\"rid\":\"good\",\"il\":1,\"ol\":2,\"input_ids\":[1]}"));
            var stride = JavaLoadClient.class.getDeclaredField("tokenStride"); stride.setAccessible(true);
            assertEquals(2L, stride.getLong(client));
            var a = client.parseTraceRecord(JSON.readTree("{\"rid\":\"a\",\"il\":1024,\"ol\":2,\"bh\":[1,2]}"));
            var b = client.parseTraceRecord(JSON.readTree("{\"rid\":\"b\",\"il\":1024,\"ol\":2,\"bh\":[1,3]}"));
            var nextA = client.makeLoopRequest(a, 1, 0);
            var nextB = client.makeLoopRequest(b, 1, 1);
            assertNotEquals(a.blockKeys, nextA.blockKeys);
            assertNotEquals(nextA.blockKeys, nextB.blockKeys);
            assertEquals(nextA.blockKeys.get(0), nextB.blockKeys.get(0));
        }
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
