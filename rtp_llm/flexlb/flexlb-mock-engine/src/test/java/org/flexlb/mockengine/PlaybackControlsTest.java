package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.*;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class PlaybackControlsTest {
    @TempDir Path temp;
    static final ObjectMapper JSON=new ObjectMapper();
    static Playback policy(String... pairs) {
        Map<String,String> env=new HashMap<>(Map.of("SEND_MODE","uniform","PLAYBACK_SEED","42"));
        for(int i=0;i<pairs.length;i+=2) env.put(pairs[i],pairs[i+1]);
        return new Playback(env,false,30);
    }
    @Test void exactCurveAndReplayWarpConserveIntegratedIntensity() {
        Playback p=policy("RATE_CURVE","[[0,1],[10,3],[20,1]]");
        Playback.Clock clock=p.new Clock();
        assertEquals(10,clock.due(2000,100,0),1e-12);
        assertEquals(20,clock.due(4000,100,0),1e-12);
        assertEquals(30,clock.due(5000,100,0),1e-12);
        assertEquals(10,p.controls.inverse(20),1e-12); // raw replay at 20s -> wall 10s
        assertEquals(20,p.controls.inverse(40),1e-12);
        for(int i=0;i<1000;i++) {
            double due=p.new Clock().due(i,100,0);
            assertEquals(i,100*(due+.1*due*due),1e-9);
        }
    }
    @Test void poissonIsSeededAndShardsMergeToIdenticalGlobalSchedule() {
        Playback p=policy("ARRIVAL_PROCESS","poisson","RATE_CURVE","[[0,1],[10,3],[20,1]]");
        Playback.Clock full=p.new Clock();
        Playback.Clock[] shards={p.new Clock(),p.new Clock(),p.new Clock()};
        int count=0; double previous=-1;
        for(int i=0;i<10000;i++) {
            double due=full.due(i,100,0);
            assertEquals(due,shards[i%3].due(i,100,0),0);
            assertTrue(due>previous); previous=due;
            if(due<30) count++;
        }
        assertTrue(Math.abs(count-5000)<5*Math.sqrt(5000));
        assertEquals(1.5354135069822445,policy("ARRIVAL_PROCESS","poisson").new Clock().due(0,1,0),1e-14);
        assertNotEquals(p.new Clock().due(0,100,0),policy("ARRIVAL_PROCESS","poisson","PLAYBACK_SEED","43").new Clock().due(0,100,0));
    }
    @Test void dynamicRetentionAnchorsLapZeroAndHasNestedSets() {
        List<Integer> tokens=new ArrayList<>(); List<Long> keys=new ArrayList<>();
        for(int i=0;i<10000;i++) {tokens.add(i+1);keys.add((long)i);}
        for(String schedule:List.of("{\"kind\":\"linear\",\"start\":0,\"end\":1,\"laps\":5}",
                "{\"kind\":\"sequence\",\"values\":[0,0.25,0.5,0.75,1]}")) {
            Playback p=policy("LAP_IDENTITY","partial","LAP_RETAIN_SCHEDULE",schedule);
            Set<Integer> previous=new HashSet<>();
            for(int lap=1;lap<=6;lap++) {
                var result=p.relabel(tokens,keys,1,lap,10001);
                assertEquals(result,p.relabel(tokens,keys,1,lap,10001));
                Set<Integer> retained=new HashSet<>();
                for(int i=0;i<tokens.size();i++) if(tokens.get(i).equals(result.get(i))) retained.add(i);
                assertTrue(retained.containsAll(previous));
                double expected=Math.min(1,(lap-1)/4.0)*tokens.size();
                assertTrue(Math.abs(retained.size()-expected)<200);
                previous=retained;
            }
        }
        Playback decay=policy("LAP_IDENTITY","partial","LAP_RETAIN_SCHEDULE","{\"kind\":\"sequence\",\"values\":[1,0.5,0]}");
        assertEquals(tokens,decay.relabel(tokens,keys,1,1,10001));
        assertNotEquals(tokens,decay.relabel(tokens,keys,1,3,10001));
    }
    @Test void invalidRawControlsFailLoud() {
        for(String[] invalid:List.of(new String[]{"RATE_CURVE","[[1,1]]"},new String[]{"RATE_CURVE","[[0,0]]"},
                new String[]{"RATE_CURVE","null"},new String[]{"RATE_CURVE","[[0,1]] []"},
                new String[]{"RATE_CURVE","[[0,1],[0,2]]"},new String[]{"ARRIVAL_PROCESS","typo"},
                new String[]{"ARRIVAL_PROCESS","poisson","SEND_MODE","replay"},
                new String[]{"ARRIVAL_PROCESS","poisson","PLAYBACK_SEED",""},
                new String[]{"RATE_CURVE","[[0,1]]","RAMP_UP_SECONDS","1"},
                new String[]{"LAP_IDENTITY","partial","LAP_RETAIN_SCHEDULE","{\"kind\":\"sequence\",\"values\":[0,1,0]}"},
                new String[]{"LAP_IDENTITY","partial","LAP_RETAIN_PROBABILITY","0","LAP_RETAIN_SCHEDULE","{\"kind\":\"sequence\",\"values\":[0,1]}"})) {
            assertThrows(IllegalArgumentException.class,()->policy(invalid),Arrays.toString(invalid));
        }
    }
    private JavaLoadClient.Config config(Path trace,Path out,int shards,int shard,String mode) {
        return new JavaLoadClient.Config(trace.toString(),"127.0.0.1:7001","127.0.0.1:7003",
            0,16,1.0,1,out.toString(),shards,shard,0,120000L,500.0,false,true,1,1,0L,120,true,
            "engine_service","",false,10,1000,0,0,"",false,"",true,0,0,mode,100,0,false,List.of());
    }
    @Test void actualDryRunSendRateTracksDeclaredCurveWithinFivePercent() throws Exception {
        for(String arrival:List.of("deterministic","poisson")) {
        Path trace=temp.resolve("timed.jsonl"), out=temp.resolve("timed-"+arrival);
        Files.writeString(trace,"{\"rid\":\"timed\",\"ts\":0,\"il\":1,\"ol\":1,\"priority\":50,\"cache_key_block_size\":512,\"input_token_blocks\":[1]}\n");
        var config=new JavaLoadClient.Config(trace.toString(),"127.0.0.1:7001","127.0.0.1:7003",
            2,16,1.0,1,out.toString(),1,0,0,120000L,500.0,false,true,1,1,0L,120,true,
            "engine_service","",false,10,1000,0,0,"",false,"",true,0,0,"uniform",100,0,false,List.of());
        Playback p=policy("MAX_LAPS","0","RATE_CURVE","[[0,0.5],[1,1.5],[2,0.5]]","ARRIVAL_PROCESS",arrival);
        new JavaLoadClient(config).run(p);
        List<JsonNode> actual=read(out.resolve("client_events.jsonl"));
        // Integral is 2 seconds at 100 base QPS; test actual sends, not only due times.
        Path plan=out.resolve("plan.jsonl"); PlaybackPlan.write(out.resolve("playback.json"),trace,plan);
        int planned=read(plan).size();
        assertEquals(200,planned,arrival.equals("poisson") ? 5*Math.sqrt(200) : 1);
        assertEquals(planned,actual.size(),Math.ceil(planned*0.05));
        long firstDue=actual.stream().mapToLong(row->row.path("send_due_epoch_ms").asLong()).min().orElseThrow();
        long inside=actual.stream().filter(row->{long t=row.path("send_start_epoch_ms").asLong()-firstDue;return t>=0 && t<2000;}).count();
        assertEquals(planned,inside,Math.ceil(planned*0.05));
        double maxLag=actual.stream().mapToDouble(row->row.path("pacing_lag_ms").asDouble()).max().orElseThrow();
        assertTrue(maxLag<100,"dry-run pacing lag: "+maxLag);
        }
    }
    private List<JsonNode> read(Path path) throws Exception {
        List<JsonNode> out=new ArrayList<>();
        for(String line:Files.readAllLines(path)) out.add(JSON.readTree(line));
        return out;
    }
    @Test void dryRunSnapshotReconstructsUnevenLapShardsAndReplay() throws Exception {
        Path trace=temp.resolve("trace.jsonl");
        StringBuilder source=new StringBuilder();
        for(int i=0;i<5;i++) source.append("{\"rid\":\"r").append(i)
            .append("\",\"ts\":").append(i*10).append(",\"il\":1,\"ol\":4,\"priority\":50,\"cache_key_block_size\":512,\"input_token_blocks\":[1]}\n");
        Files.writeString(trace,source);
        for(String mode:List.of("uniform","replay","legacy")) {
            List<Long> merged=new ArrayList<>();
            for(int shard=0;shard<3;shard++) {
                Path out=temp.resolve(mode+shard);
                Playback p=policy("SEND_MODE",mode.equals("legacy")?"uniform":mode,"MAX_LAPS","3","RATE_CURVE","[[0,1],[1,2]]",
                    "ARRIVAL_PROCESS",mode.equals("uniform")?"poisson":"deterministic",
                    "LAP_IDENTITY","partial","LAP_RETAIN_SCHEDULE","{\"kind\":\"sequence\",\"values\":[0,1]}");
                if(mode.equals("legacy")) p=policy("MAX_LAPS","3");
                new JavaLoadClient(config(trace,out,3,shard,mode.equals("legacy")?"uniform":mode)).run(p);
                Path plan=out.resolve("plan.jsonl");
                PlaybackPlan.write(out.resolve("playback.json"),trace,plan);
                List<JsonNode> expected=read(plan),actual=read(out.resolve("client_events.jsonl"));
                assertEquals(mode.equals("legacy") ? (shard==2?3:6) : 5,expected.size()); assertEquals(expected.size(),actual.size());
                actual.sort(Comparator.comparingDouble(r->r.path("send_due_epoch_ms").asDouble()));
                double origin=actual.get(0).path("send_due_epoch_ms").asDouble()-expected.get(0).path("due_seconds").asDouble()*1000;
                for(int i=0;i<expected.size();i++) {
                    JsonNode e=expected.get(i),a=actual.get(i);
                    merged.add(e.path("global_index").asLong());
                    assertEquals(e.path("rid").asText(),a.path("rid").asText());
                    assertEquals(e.path("ol").asInt(),a.path("output_len").asInt());
                    assertEquals(e.path("due_seconds").asDouble(),(a.path("send_due_epoch_ms").asDouble()-origin)/1000,0.0011);
                }
                Path again=out.resolve("again.jsonl");PlaybackPlan.write(out.resolve("playback.json"),trace,again);
                assertEquals(Files.readString(plan),Files.readString(again));
                // A changed source cannot be silently replayed from an old snapshot.
                Path wrong=temp.resolve("wrong");Files.writeString(wrong,source+"\n");
                assertThrows(IllegalArgumentException.class,()->PlaybackPlan.write(out.resolve("playback.json"),wrong,again));
            }
            Collections.sort(merged);
            if(!mode.equals("legacy")) for(int i=0;i<15;i++) assertEquals(i,merged.get(i));
        }
    }
}
