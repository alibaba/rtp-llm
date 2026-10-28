import org.flexlb.balance.scheduler.PrefillQueueShedding;
import org.flexlb.balance.scheduler.PrefillTrafficTrend;
import org.flexlb.config.PrefillQueueSheddingConfig;

import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/** Deterministic policy experiment with finite worker caches and completion-dependent turns.
 * This is not a Master, network or GPU benchmark. Run the real Master traffic suite as well.
 */
public class PrefillOverloadFeedback {
    static final int WORKERS = 4;
    static final long SOFT = 3000, HARD = 6000, END = 120_000;
    record Request(long id, long at, long session, int turn, long tokens, long hit, long prediction) { }
    record Event(long at, long sequence, int kind, Request request, int worker) { }
    record Cache(long tokens, long at) { }
    static final class Run {
        final boolean bounded, conversation;
        final double overload, bias;
        final Random arrivals, shapes;
        final long ttl;
        final PrefillQueueSheddingConfig config = new PrefillQueueSheddingConfig();
        final PrefillTrafficTrend trend = new PrefillTrafficTrend(config.getRecovery());
        final LinkedHashMap<Long, Request> queue = new LinkedHashMap<>();
        final List<LinkedHashMap<Long, Cache>> caches = new ArrayList<>();
        final boolean[] busy = new boolean[WORKERS];
        final PriorityQueue<Event> events = new PriorityQueue<>(Comparator.comparingLong(Event::at).thenComparingLong(Event::sequence));
        long now, ids, sequence, offered, completed, rejected, expired, hitTokens, inputTokens, busyMs;
        int maxQueue;
        Run(boolean bounded, boolean conversation, double overload, double bias, long ttl, int seed) {
            this.bounded=bounded;this.conversation=conversation;this.overload=overload;this.bias=bias;this.ttl=ttl;
            arrivals=new Random(seed);shapes=new Random(seed ^ 0x1234);
            config.setExpectedWaitThresholdMs(SOFT);config.setMaxWaitMs(HARD);
            config.getRecovery().setMaxEvictionFraction(Double.parseDouble(System.getProperty("flexlb.feedback.evictionFraction", "0.1")));
            for(int w=0;w<WORKERS;w++) caches.add(new LinkedHashMap<>());
        }
        void event(long at,int kind,Request r,int w){events.add(new Event(at,++sequence,kind,r,w));}
        long hit(int w,long session,long tokens){
            Cache c=caches.get(w).get(session);
            return c==null || now-c.at>ttl ? 0 : Math.min(c.tokens,tokens-700);
        }
        void arrive(long session,int turn,long tokens){
            long best=0;
            for(int w=0;w<WORKERS;w++)best=Math.max(best,hit(w,session,tokens));
            Request r=new Request(++ids,now,session,turn,tokens,best,Math.max(1,Math.round((tokens-best)/70.0*bias)));
            offered++;trend.arrive(tokens);queue.put(r.id,r);maxQueue=Math.max(maxQueue,queue.size());dispatch();
        }
        void dispatch(){
            for(int w=0;w<WORKERS && !queue.isEmpty();w++) if(!busy[w]){
                var it=queue.values().iterator();var r=it.next();it.remove();
                if(now-r.at>=HARD){expired++;trend.discard((double)r.prediction/WORKERS);w--;continue;}
                // Prefer a cache owner if it is free; never invent cache on another worker.
                int target=w;long h=hit(w,r.session,r.tokens);
                for(int j=w+1;j<WORKERS;j++)if(!busy[j] && hit(j,r.session,r.tokens)>h){target=j;h=hit(j,r.session,r.tokens);}
                long service=Math.max(1,(r.tokens-h+69)/70);
                busy[target]=true;busyMs+=Math.min(service,END-now);
                event(now+service,1,new Request(r.id,r.at,r.session,r.turn,r.tokens,h,0),target);
                if(target!=w)w--;
            }
        }
        void maintain(){
            var it=queue.values().iterator();
            while(it.hasNext()){var r=it.next();if(now-r.at>=HARD){it.remove();expired++;trend.discard((double)r.prediction/WORKERS);}}
            var candidates=queue.values().stream().map(r->new PrefillQueueShedding.Candidate<>(r,50,r.id,
                    new PrefillQueueShedding.Estimate(r.tokens,r.hit,r.prediction,WORKERS))).toList();
            double work=candidates.stream().mapToDouble(c->c.estimate().fleetWorkMs()).sum();
            var snapshot=trend.observe(now,work,SOFT,HARD,
                    candidates.stream().mapToLong(c->c.estimate().workMs()).max().orElse(0));
            List<Request> victims;
            if(bounded){victims=PrefillQueueShedding.adaptiveVictims(candidates,config,snapshot,r->now-r.at);}
            else {
                victims=PrefillQueueShedding.victims(candidates,snapshot.grace()?HARD:SOFT).stream()
                        .filter(r->work>=HARD || now-r.at>=SOFT).toList();
            }
            for(var r:victims){if(queue.remove(r.id)==null)throw new AssertionError("duplicate eviction");rejected++;trend.discard((double)r.prediction/WORKERS);}
            dispatch();event(now+100,2,null,0);
        }
        void run(PrintWriter out,int seed){
            event(0,0,null,0);event(100,2,null,0);
            while(!events.isEmpty()){
                var e=events.poll();now=e.at;if(now>=END)break;
                if(e.kind==0){
                    long id=shapes.nextDouble()<.2 ? 1_000_000+ids : shapes.nextInt(24);
                    arrive(id,0,7000);
                    double load=now<20_000?.5:now<22_000?.02:now<40_000?2:now<90_000?overload:.2;
                    double qps=140*load/(conversation?8:1);
                    event(now+Math.max(1,Math.round(-Math.log1p(-arrivals.nextDouble())*1000/qps)),0,null,0);
                } else if(e.kind==1){
                    busy[e.worker]=false;var r=e.request;completed++;hitTokens+=r.hit;inputTokens+=r.tokens;
                    var cache=caches.get(e.worker);cache.remove(r.session);cache.put(r.session,new Cache(r.tokens,now));
                    while(cache.size()>16)cache.remove(cache.keySet().iterator().next());
                    if(conversation && r.turn<7)event(now+50,3,r,0);
                    dispatch();
                } else if(e.kind==2){maintain();}
                else {var r=e.request;arrive(r.session,r.turn+1,r.tokens+700);}
            }
            long active=0;for(boolean b:busy)if(b)active++;
            if(offered!=completed+rejected+expired+queue.size()+active)throw new AssertionError("request conservation");
            out.printf(Locale.ROOT,"%s,%s,%d,%.1f,%.1f,%d,%d,%d,%d,%d,%.6f,%.6f,%d%n",
                    bounded?"bounded":"baseline",conversation?"conversation":"replay",seed,overload,bias,ttl,
                    offered,completed,rejected,expired,inputTokens==0?0:(double)hitTokens/inputTokens,
                    (double)busyMs/(END*WORKERS),maxQueue);
        }
    }
    public static void main(String[] args)throws Exception{
        try(var out=new PrintWriter(Files.newBufferedWriter(Path.of(args[0])))){
            out.println("policy,scenario,seed,overload,prediction_bias,cache_ttl_ms,offered,completed,rejected,expired,hit_rate,utilization,max_queue");
            for(boolean conversation:new boolean[]{false,true})for(double load:new double[]{2,5,10})
                for(double bias:new double[]{1,10,100})for(long ttl:new long[]{2000,20000})for(int seed=1;seed<=3;seed++)
                    for(boolean bounded:new boolean[]{false,true})new Run(bounded,conversation,load,bias,ttl,seed).run(out,seed);
        }
    }
}
