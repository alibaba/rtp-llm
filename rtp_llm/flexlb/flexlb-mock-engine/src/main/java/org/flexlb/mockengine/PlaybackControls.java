package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;

/** Opt-in, versioned planning controls. No wall clock or per-shard randomness. */
final class PlaybackControls {
    private static final ObjectMapper JSON = new ObjectMapper();
    final JsonNode curve, schedule;
    final String arrival;
    final boolean explicitArrival;

    PlaybackControls(Map<String,String> env, String identity) {
        curve = parse(env, "RATE_CURVE");
        schedule = parse(env, "LAP_RETAIN_SCHEDULE");
        explicitArrival = present(env,"ARRIVAL_PROCESS");
        arrival = explicitArrival ? env.get("ARRIVAL_PROCESS") : "deterministic";
        require(arrival.equals("deterministic") || arrival.equals("poisson"), "invalid ARRIVAL_PROCESS");
        if (arrival.equals("poisson") || schedule != null)
            require(present(env,"PLAYBACK_SEED"), "new random controls require explicit PLAYBACK_SEED");
        if (arrival.equals("poisson"))
            require("uniform".equals(env.get("SEND_MODE")), "poisson requires uniform SEND_MODE");
        if (curve != null || arrival.equals("poisson")) {
            require(defaultNumber(env,"BURST_FACTOR",1)==1 && defaultNumber(env,"DIURNAL_AMPLITUDE",0)==0
                && defaultNumber(env,"RAMP_UP_SECONDS",0)==0 && !Set.of("true","1","yes").contains(env.getOrDefault("GRADIENT", "").toLowerCase(java.util.Locale.ROOT)),
                "rate_curve/poisson excludes legacy burst, wave, ramp and GRADIENT");
            require(Set.of("uniform","replay").contains(env.getOrDefault("SEND_MODE","replay")), "invalid SEND_MODE");
        }
        if (curve != null) {
            require(curve.isArray() && curve.size()>0 && curve.size()<=4096, "invalid RATE_CURVE size");
            double previous = -1;
            for (JsonNode point : curve) {
                require(point.isArray() && point.size()==2, "invalid RATE_CURVE point");
                double t = number(point.get(0),0,1e9), v = number(point.get(1),1e-6,1e6);
                require(t>previous, "RATE_CURVE times must increase");
                previous=t;
            }
            require(curve.get(0).get(0).doubleValue()==0, "RATE_CURVE must start at zero");
        }
        if (schedule != null) {
            require(identity.equals("partial") && !present(env,"LAP_RETAIN_PROBABILITY"),
                    "retain_schedule requires partial and excludes scalar retain probability");
            String kind=schedule.path("kind").asText();
            if (kind.equals("linear")) {
                fields(schedule,Set.of("kind","start","end","laps"));
                number(schedule.get("start"),0,1); number(schedule.get("end"),0,1);
                require(schedule.get("laps").isIntegralNumber() && schedule.get("laps").canConvertToInt()
                    && schedule.get("laps").intValue()>=2, "invalid linear laps");
            } else if (kind.equals("sequence")) {
                fields(schedule,Set.of("kind","values"));
                JsonNode values=schedule.get("values");
                require(values.isArray() && values.size()>0 && values.size()<=4096,"invalid retention sequence");
                boolean up=true, down=true; double previous=number(values.get(0),0,1);
                for (JsonNode value:values) {
                    double v=number(value,0,1); up &= v>=previous; down &= v<=previous; previous=v;
                }
                require(up || down,"retention sequence must be monotonic");
            } else throw new IllegalArgumentException("unknown retain_schedule kind");
        }
    }

    private static boolean present(Map<String,String> env,String key) {
        return env.get(key)!=null && !env.get(key).isBlank();
    }
    private static JsonNode parse(Map<String,String> env,String key) {
        if (!present(env,key)) return null;
        try {
            JsonNode node=JSON.reader().with(com.fasterxml.jackson.databind.DeserializationFeature.FAIL_ON_TRAILING_TOKENS)
                    .readTree(env.get(key));
            require(node!=null && !node.isNull(),"null " + key);
            return node;
        } catch (java.io.IOException e) { throw new IllegalArgumentException("invalid " + key,e); }
    }
    private static double defaultNumber(Map<String,String> env,String key,double fallback) {
        return present(env,key) ? Double.parseDouble(env.get(key)) : fallback;
    }
    private static void fields(JsonNode node,Set<String> expected) {
        Set<String> actual=new HashSet<>(); node.fieldNames().forEachRemaining(actual::add);
        require(node.isObject() && actual.equals(expected),"invalid retain_schedule fields");
    }
    private static double number(JsonNode node,double low,double high) {
        require(node!=null && node.isNumber() && Double.isFinite(node.doubleValue())
                && node.doubleValue()>=low && node.doubleValue()<=high,"invalid playback number");
        return node.doubleValue();
    }
    private static void require(boolean condition,String message) {
        if (!condition) throw new IllegalArgumentException(message);
    }
    boolean enabled() { return curve!=null || schedule!=null || explicitArrival; }
    void snapshot(Map<String,Object> out) {
        if (curve!=null) out.put("rate_curve",curve);
        if (schedule!=null) out.put("retain_schedule",schedule);
        if (explicitArrival) out.put("arrival",arrival);
        if (enabled()) out.put("planner_version","PLAYBACK_CONTROLS_V1");
    }
    double retention(int lap) {
        if (schedule.path("kind").asText().equals("sequence")) {
            JsonNode values=schedule.get("values");
            return values.get(Math.min(lap-1,values.size()-1)).doubleValue();
        }
        double start=schedule.get("start").doubleValue(), end=schedule.get("end").doubleValue();
        return start+(end-start)*Math.min(1,(lap-1.0)/(schedule.get("laps").intValue()-1.0));
    }
    /** Exact integral inversion, with a stable quadratic solution on each segment. */
    double inverse(double area) {
        if (curve==null) return area;
        for (int i=1;i<curve.size();i++) {
            double left=curve.get(i-1).get(0).doubleValue(), right=curve.get(i).get(0).doubleValue();
            double a=curve.get(i-1).get(1).doubleValue(), b=curve.get(i).get(1).doubleValue();
            double segment=(right-left)*(a+b)/2;
            if (area<=segment) {
                double slope=(b-a)/(right-left);
                return left+2*area/(a+Math.sqrt(Math.max(0,a*a+2*slope*area)));
            }
            area-=segment;
        }
        JsonNode last=curve.get(curve.size()-1);
        return last.get(0).doubleValue()+area/last.get(1).doubleValue();
    }
}
