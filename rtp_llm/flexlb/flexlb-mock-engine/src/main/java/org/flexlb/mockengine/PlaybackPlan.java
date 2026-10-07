package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/** Offline semantic reconstruction from PLAYBACK_CONTROLS_V1 snapshot + pinned canonical trace. */
public final class PlaybackPlan {
    private static final ObjectMapper JSON = new ObjectMapper();
    private PlaybackPlan() {}

    static String sha256(Path path) throws Exception {
        var digest=java.security.MessageDigest.getInstance("SHA-256");
        try (var input=Files.newInputStream(path)) {
            byte[] buffer=new byte[65536]; int read;
            while ((read=input.read(buffer))!=-1) digest.update(buffer,0,read);
        }
        return HexFormat.of().formatHex(digest.digest());
    }

    static Map<String,String> environment(JsonNode snapshot) {
        if (!Set.of("PLAYBACK_CONTROLS_V1","PLAYBACK_LEGACY_V1").contains(snapshot.path("planner_version").asText()))
            throw new IllegalArgumentException("unsupported playback planner version");
        Map<String,String> env=new HashMap<>();
        String[][] fields={{"mode","SEND_MODE"},{"max_laps","MAX_LAPS"},{"identity","LAP_IDENTITY"},
            {"seed","PLAYBACK_SEED"},{"retain_probability","LAP_RETAIN_PROBABILITY"},
            {"rate_curve","RATE_CURVE"},{"arrival","ARRIVAL_PROCESS"},{"retain_schedule","LAP_RETAIN_SCHEDULE"},
            {"burst_factor","BURST_FACTOR"},{"burst_period_seconds","BURST_PERIOD_SECONDS"},
            {"burst_duty","BURST_DUTY"},{"diurnal_amplitude","DIURNAL_AMPLITUDE"},
            {"diurnal_period_seconds","DIURNAL_PERIOD_SECONDS"},{"ramp_up_seconds","RAMP_UP_SECONDS"}};
        for (String[] field:fields) if(snapshot.has(field[0])) {
            JsonNode node=snapshot.get(field[0]);
            env.put(field[1],node.isContainerNode()?node.toString():node.asText());
        }
        if(snapshot.has("retain_schedule")) env.remove("LAP_RETAIN_PROBABILITY");
        return env;
    }

    static void write(Path snapshotPath,Path trace,Path output) throws Exception {
        if (output.toAbsolutePath().normalize().equals(trace.toAbsolutePath().normalize())
                || output.toAbsolutePath().normalize().equals(snapshotPath.toAbsolutePath().normalize()))
            throw new IllegalArgumentException("plan output must not overwrite source evidence");
        JsonNode snapshot=JSON.readTree(snapshotPath.toFile());
        if(!snapshot.path("offline_plan_supported").asBoolean(false))
            throw new IllegalArgumentException("snapshot uses unsupported runtime overrides");
        if (!sha256(trace).equals(snapshot.path("trace_sha256").asText()))
            throw new IllegalArgumentException("trace SHA does not match playback snapshot");
        int duration=snapshot.path("duration_seconds").asInt(), shards=snapshot.path("num_shards").asInt();
        int shard=snapshot.path("shard_index").asInt(), limit=snapshot.path("limit").asInt();
        if(shards<1 || shard<0 || shard>=shards) throw new IllegalArgumentException("invalid snapshot shard");
        Playback policy=new Playback(environment(snapshot),false,duration);
        List<JsonNode> rows=new ArrayList<>();
        List<List<Integer>> tokens=new ArrayList<>(); List<List<Long>> keys=new ArrayList<>();
        long stride=snapshot.path("token_stride").asLong();
        if(stride<1) throw new IllegalArgumentException("missing token stride");
        try(var reader=Files.newBufferedReader(trace)) {
            String line;
            while((line=reader.readLine())!=null) if(!line.isBlank()) rows.add(JSON.readTree(line));
        }
        rows.sort(Comparator.comparingLong(row->row.path("ts").asLong()));
        if(rows.isEmpty()) throw new IllegalArgumentException("empty trace");
        for(JsonNode row:rows) {
            if(!row.has("rid") || (!row.has("input_ids") && !row.has("input_token_blocks")))
                throw new IllegalArgumentException("offline planner requires canonical token-backed trace");
            List<Integer> ids=JavaLoadClient.decodeInputTokens(row,row.path("il").asInt());
            tokens.add(ids); keys.add(JavaLoadClient.computeBlockKeys(ids,row.path("cache_key_block_size").asInt()));
            for(int value:ids) if(value>=stride) throw new IllegalArgumentException("invalid token stride");
        }
        long first=rows.get(0).path("ts").asLong();
        long span=Math.max(1,rows.get(rows.size()-1).path("ts").asLong()-first)+1;
        if(!policy.controls.enabled() && shard>=rows.size())
            throw new IllegalArgumentException("empty legacy shard");
        Playback.Clock clock=policy.new Clock();
        long sent=0;
        try(var writer=Files.newBufferedWriter(output)) {
            for(int lap=0;;lap++) {
                for(int i=0;i<rows.size();i++) {
                    long index;
                    if(policy.controls.enabled()) {
                        index=(long)lap*rows.size()+i;
                        if(limit>0 && index>=limit) return;
                        if(index%shards!=shard) continue;
                    } else {
                        if(policy.maxLaps==1 && limit>0 && i>=limit) return;
                        if(i%shards!=shard) continue;
                        if(limit>0 && sent>=limit) return;
                        index=sent*shards+shard;
                    }
                    JsonNode row=rows.get(i);
                    double due=snapshot.path("mode").asText().equals("uniform")
                        ? clock.due(index,snapshot.path("qps").asDouble(),snapshot.path("ramp_up_seconds").asDouble())
                        : snapshot.path("speed").asDouble()<=0 ? 0
                        : policy.controls.inverse((row.path("ts").asLong()-first+lap*span)/1000.0/snapshot.path("speed").asDouble());
                    if(duration>0 && due>=duration) return;
                    List<Integer> ids=policy.relabel(tokens.get(i),keys.get(i),row.path("cache_key_block_size").asInt(),lap,stride);
                    ObjectNode plan=JSON.createObjectNode();
                    plan.put("global_index",index); plan.put("due_seconds",due); plan.put("iteration",lap);
                    plan.put("rid",row.path("rid").asText()+(lap==0?"":"_S"+shard+"_L"+lap));
                    plan.put("il",row.path("il").asInt()); plan.put("ol",row.path("ol").asInt());
                    plan.set("block_keys",JSON.valueToTree(JavaLoadClient.computeBlockKeys(ids,row.path("cache_key_block_size").asInt())));
                    writer.write(plan.toString()); writer.newLine(); sent++;
                }
                if(!policy.more(lap)) return;
            }
        }
    }
    public static void main(String[] args) throws Exception {
        if(args.length!=3) throw new IllegalArgumentException("Usage: PlaybackPlan playback.json trace.jsonl plan.jsonl");
        write(Path.of(args[0]),Path.of(args[1]),Path.of(args[2]));
    }
}
