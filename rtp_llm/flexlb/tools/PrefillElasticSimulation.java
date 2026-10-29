import org.flexlb.balance.strategy.PrefillTokenBalance;
import org.flexlb.balance.strategy.PrefillTokenWorkload;
import org.flexlb.balance.scheduler.PrefillQueueShedding;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Deterministic CPU experiment. Real policy primitives; explicit per-worker caches and serial linear service.
 * No GPU, decode, network, KV capacity eviction or engine status lag. Run with Java 21 and compiled sync/common.
 */
public class PrefillElasticSimulation {
    static final int N = 150, INITIAL = 100, ROUNDS = 21;
    static final double HIT = Double.parseDouble(System.getProperty("flexlb.sim.hit", "0.9"));
    static final double TPS = 70_000, TTL = 1_200_000, MAX_WAIT = 60_000, SHED_WAIT = 30_000;
    static final double[] LOADS = Arrays.stream(System.getProperty("flexlb.sim.loads", "0.2,0.5,0.8,1.2,1.5,1.5").split(",")).mapToDouble(Double::parseDouble).toArray();
    static final double GROWTH = growth(), MEAN_FACTOR = sumGrowth() / ROUNDS;
    enum Policy { SHORT_TTFT, SHORT_TTFT_CAPPED, CACHE_ONLY, CACHE_WINDOW, ELASTIC, ADAPTIVE, SESSION }
    enum Scene { REPLAY, CONVERSATION, COLD_BURST, TTL_GAP, WARM_REPLAY, HOTSPOT, COLD_BURST_OPEN, SCALE_OUT, HOTSPOT_SCALE_OUT, RECOVERY_BURST, RAMP, SCALE_IN, SCALE_CYCLE }
    record Cache(long tokens, double touched) { }
    static final class Query {
        final long id, session;
        final int turn;
        final double at, base, rootAt;
        long tokens, prefix;
        long reserved, hit, actualWork;
        int worker = -1, migratedFrom = -1;
        boolean escaping;
        final int priority;
        Query(long id, long session, int turn, double at, double base, double rootAt) {
            this.id=id; this.session=session; this.turn=turn; this.at=at; this.base=base; this.rootAt=rootAt;
            tokens=Math.max(1, Math.round(base * Math.pow(GROWTH, turn)));
            int bucket = Math.floorMod(Long.hashCode(session * 0x9E3779B97F4A7C15L), 100);
            priority = bucket < 20 ? 10 : bucket < 80 ? 50 : 90;
            prefix=turn==0 ? 0 : Math.min(tokens-1, Math.round(base * Math.pow(GROWTH, turn-1)));
        }
    }
    record Event(double at, long sequence, int kind, Query query, int worker, List<Query> batch) { }
    static final class Worker {
        final Map<Long, Cache> cache = new HashMap<>();
        final ArrayDeque<Query> queue = new ArrayDeque<>();
        final PrefillTokenWorkload work;
        double busyUntil, pendingCompute, collectionDue;
        boolean collecting, running;
        long input, hit, compute, useful, count;
        // window remains only as a legacy experiment/CSV label; routing has no history window.
        Worker(long window) { work=new PrefillTokenWorkload(); }
    }
    static final class Counters {
        long arrivals, completed, dropped, expired, coldArrivals, coldCompleted, coldDropped, coldExpired;
        double capacityTokens, input, theoretical, routedInput, routedHit, completedInput, completedHit, completedCompute, expiredCompute, completedUseful;
        Counters copy() { var c=new Counters(); c.capacityTokens=capacityTokens; c.arrivals=arrivals;c.completed=completed;c.dropped=dropped;c.expired=expired;
            c.coldArrivals=coldArrivals;c.coldCompleted=coldCompleted;c.coldDropped=coldDropped;c.coldExpired=coldExpired;
            c.input=input;c.theoretical=theoretical;c.routedInput=routedInput;c.routedHit=routedHit;
            c.completedInput=completedInput;c.completedHit=completedHit;c.completedCompute=completedCompute;c.expiredCompute=expiredCompute;c.completedUseful=completedUseful;return c; }
    }
    static final class Run {
        final Policy policy; final Scene scene; final int seed; final double stageMs, endMs; final long window, cap;
        final boolean reorder;
        final Worker[] workers=new Worker[N];
        final Map<Long, Map<Integer, Cache>> cacheIndex=new HashMap<>();
        final Map<Long, Integer> sessionWorkers=new HashMap<>();
        final double[] workScores=new double[N];
        final long[] pendingTokens=new long[N];
        double lastShedding=-100;
        final Random roots, lengths;
        final long[] corpus = new long[50_000];
        final PriorityQueue<Event> events=new PriorityQueue<>(Comparator.comparingDouble(Event::at).thenComparingLong(Event::sequence));
        final NavigableMap<Long, Query> waiting=new TreeMap<>();
        final Counters c=new Counters();
        final ArrayList<Double> latencies=new ArrayList<>(), timelineLatencies=new ArrayList<>(), waits=new ArrayList<>();
        final double[] hits=new double[N], scores=new double[N];
        final PrintWriter summary, timeline, perWorker;
        long eventId, requestId, sessionId, traceHash, maxQueue;
        int idleCount;
        int cursor, active=INITIAL; double now;
        final long rescueLimit=Long.getLong("flexlb.sim.rescueLimit", 32768L);
        final long rescueWait=Long.getLong("flexlb.sim.rescueWaitMs", 5000L);
        long rescued, localMigrations, graceTicks;
        final long[] droppedPriority = new long[3], completedPriority = new long[3];
        final org.flexlb.config.PrefillQueueSheddingConfig sheddingConfig = new org.flexlb.config.PrefillQueueSheddingConfig();
        final org.flexlb.balance.scheduler.PrefillTrafficTrend trend = new org.flexlb.balance.scheduler.PrefillTrafficTrend(
                new org.flexlb.config.PrefillQueueSheddingConfig.RecoveryConfig());
        long[] prevCompute=new long[N], prevInput=new long[N], prevUseful=new long[N];
        Counters prev=new Counters(), measured; long[] measuredCompute, measuredInput, measuredUseful;
        Run(Policy policy, Scene scene,int seed,double stageSeconds,long window,long cap,boolean reorder,
            PrintWriter summary, PrintWriter timeline, PrintWriter perWorker) {
            this.policy=policy;this.scene=scene;this.seed=seed;this.stageMs=stageSeconds*1000;
            endMs=stageMs*LOADS.length;this.window=window;this.cap=cap;this.reorder=reorder;
            sheddingConfig.setRecovery(new org.flexlb.config.PrefillQueueSheddingConfig.RecoveryConfig());
            this.summary=summary;this.timeline=timeline;this.perWorker=perWorker;
            roots=new Random(seed); lengths=new Random(seed^0x187abcd);
            for(int i=0;i<N;i++)workers[i]=new Worker(window);
            if(independent()) {
                var random=new Random(seed^0x347abe);double total=0;
                for(int i=0;i<corpus.length;i++){corpus[i]=Math.max(1,Math.round(100_000*Math.min(20,Math.exp(.9*random.nextGaussian()-.405))));total+=corpus[i];}
                for(int i=0;i<corpus.length;i++) {
                    corpus[i]=Math.max(10,10*Math.round(corpus[i]*100_000.0*corpus.length/total/10));
                    int home=(scene==Scene.HOTSPOT || scene==Scene.HOTSPOT_SCALE_OUT) && i%5!=0 ? i%10 : i%INITIAL;
                    Cache initial=new Cache(Math.round(HIT*corpus[i]),0);
                    workers[home].cache.put((long)i+1,initial);
                    cacheIndex.computeIfAbsent((long)i+1,unused->new HashMap<>()).put(home,initial);
                }
            }
        }
        boolean independent(){return true;}
        boolean windowPolicy(){return policy==Policy.CACHE_WINDOW || policy==Policy.ELASTIC || policy==Policy.ADAPTIVE || policy==Policy.SESSION;}
        boolean elastic(){return policy==Policy.ELASTIC || policy==Policy.ADAPTIVE || policy==Policy.SESSION;}
        int priorityIndex(Query q){return q.priority==10 ? 0 : q.priority==50 ? 1 : 2;}
        void event(double at,int kind,Query q,int worker,List<Query> batch) {
            events.add(new Event(at,++eventId,kind,q,worker,batch));
        }
        double nextRoot() {
            double hazard=-Math.log1p(-roots.nextDouble()), at=now;
            while(at<endMs) {
                int stage=Math.min(LOADS.length-1,(int)(at/stageMs));
                double rate=LOADS[stage]*(INITIAL*TPS/(100_000*(1-HIT)))/(independent()?1:ROUNDS)/1000, boundary=(stage+1)*stageMs;
                if(hazard<(boundary-at)*rate)return at+hazard/rate;
                hazard-=(boundary-at)*rate;at=boundary;
            }return endMs+1;
        }
        double base() {
            // Lognormal: many short queries, few long ones; bounded at ~2M aggregate-mean tokens.
            // sigma .9; clipping at 20x has negligible (<0.1%) mean impact.
            return 100_000/MEAN_FACTOR*Math.min(20, Math.exp(.9*lengths.nextGaussian()-.405));
        }
        double think(long session,int turn) {
            // Independent of event ordering and routing policy, so paired traces stay comparable.
            var random=new Random(session*0x9E3779B97F4A7C15L+turn*7919L+seed);
            double delay=2000+random.nextDouble()*4000;
            if(scene==Scene.TTL_GAP && turn==10 && session%5==0)delay+=1_300_000;
            return delay;
        }
        void arrive(Query q) {
            trend.arrive(q.tokens);
            c.arrivals++;if(q.session<0)c.coldArrivals++;c.input+=q.tokens;c.theoretical+=q.prefix;
            waiting.put(q.id,q);maxQueue=Math.max(maxQueue,waiting.size());
        }
        long hit(int worker,Query q) {
            Cache cache=workers[worker].cache.get(q.session);
            if(cache==null)return 0;
            if(now-cache.touched>=TTL) { workers[worker].cache.remove(q.session);return 0; }
            return Math.min(q.prefix,cache.tokens);
        }
        int choose(Query q) {
            long max=0;
            Arrays.fill(hits,0);
            var cached=cacheIndex.get(q.session);
            if(cached!=null)for(var entry:cached.entrySet()) {
                Cache cache=entry.getValue();
                if(entry.getKey()<active && now-cache.touched<TTL) {
                    long matched=Math.min(q.prefix,cache.tokens);
                    hits[entry.getKey()]=matched;max=Math.max(max,matched);
                }
            }
            double best=Double.POSITIVE_INFINITY;
            boolean affinity=PrefillTokenBalance.cacheFirst(max,q.tokens,50);
            int selected=-1;
            Integer sticky = policy==Policy.SESSION ? sessionWorkers.get(q.session) : null;
            if(sticky!=null && sticky>=active){sessionWorkers.remove(q.session);sticky=null;}
            if(sticky!=null && q.migratedFrom<0) {
                if(pendingTokens[sticky]<cap)return sticky;
                if(q.tokens>rescueLimit || now-q.at<rescueWait)return -1;
            }
            boolean escape = elastic() && q.migratedFrom>=0;
            if(elastic() && max>0 && q.tokens<=rescueLimit && now-q.at>=rescueWait) {
                escape=true;
                for(int i:cached.keySet())if(i<active && hits[i]==max && pendingTokens[i]<cap){escape=false;break;}
            }
            if(q.migratedFrom>=0 || sticky!=null)escape=true;
            q.escaping=escape;
            if(escape && idleCount==0)return -1;
            if((windowPolicy() || policy==Policy.CACHE_ONLY) && max>0 && !escape) {
                for(int i:cached.keySet()) {
                    if(i>=active || pendingTokens[i]>=cap || !PrefillTokenBalance.eligible((long)hits[i],max,affinity))continue;
                    var w=workers[i];
                    double score=windowPolicy() ? workScores[i]+q.tokens-hits[i]
                        : Math.max(0,w.busyUntil-now)*TPS/1000+w.pendingCompute+q.tokens-hits[i];
                    if(score<best || score==best && (selected<0 || Math.floorMod(i-cursor,N)<Math.floorMod(selected-cursor,N))) {
                        best=score;selected=i;
                    }
                }
                if(selected>=0)cursor=(selected+1)%N;
                return selected;
            }
            for(int k=0;k<N;k++) {
                int i=(cursor+k)%N;var w=workers[i];
                if(i>=active || escape && (pendingTokens[i]>0 || i==q.migratedFrom))continue;
                // Same backpressure for both policies isolates ranking from admission changes.
                if(policy!=Policy.SHORT_TTFT && pendingTokens[i]>=cap){scores[i]=Double.POSITIVE_INFINITY;continue;}
                if((windowPolicy() || policy==Policy.CACHE_ONLY) && !escape && !PrefillTokenBalance.eligible((long)hits[i],max,affinity)) {
                    scores[i]=Double.POSITIVE_INFINITY;continue;
                }
                double score=!windowPolicy()
                    ? Math.max(0,w.busyUntil-now)*TPS/1000+w.pendingCompute+q.tokens-hits[i]
                    : workScores[i]+q.tokens-hits[i];
                scores[i]=score;
                if(score<best){best=score;selected=i;}
            }
            if(selected>=0)cursor=(selected+1)%N;
            return selected;
        }
        void migrateLocal() {
            if(!elastic())return;
            boolean idle=false;
            for(int i=0;i<active;i++)if(workers[i].work.snapshot().pendingTokens()==0){idle=true;break;}
            if(!idle)return;
            var candidates=new ArrayList<Query>();
            for(int i=0;i<active;i++) {
                var w=workers[i];
                if(w.work.snapshot().pendingTokens()<cap)continue;
                for(Query q:w.queue)if(q.migratedFrom<0 && q.tokens<=rescueLimit && now-q.at>=rescueWait)candidates.add(q);
            }
            candidates.sort(Comparator.comparingLong(q->q.tokens));
            int moved=0;
            for(Query q:candidates) {
                if(moved++>=4)break;
                var w=workers[q.worker];
                if(!w.queue.remove(q))throw new AssertionError("migration ownership");
                w.pendingCompute-=q.reserved;w.work.release(q.reserved);
                q.migratedFrom=q.worker;q.worker=-1;waiting.put(q.id,q);localMigrations++;
            }
        }
        void route() {
            if(now-lastShedding>=100)migrateLocal();
            // Scan all waiting requests once per 20ms scheduling round; blocked affinity cannot block cold work.
            // Optional ablation reorders only each already-present 32-item group, never defers an item.
            idleCount=0;
            for(int i=0;i<N;i++) {
                var snapshot=workers[i].work.snapshot();
                workScores[i]=snapshot.score();pendingTokens[i]=snapshot.pendingTokens();
                if(i<active && pendingTokens[i]==0)idleCount++;
            }
            var candidates=new ArrayList<>(waiting.values());
            if(reorder)for(int start=0;start<candidates.size();start+=32) {
                candidates.subList(start,Math.min(start+32,candidates.size()))
                    .sort(Comparator.comparingLong((Query q)->q.tokens-q.prefix).reversed());
            }
            for(Query q:candidates) {
                if(now-q.at>=MAX_WAIT){
                    long matched=0;var copies=cacheIndex.get(q.session);
                    if(copies!=null)for(var copy:copies.entrySet())if(copy.getKey()<active && now-copy.getValue().touched<TTL)
                        matched=Math.max(matched,Math.min(q.prefix,copy.getValue().tokens));
                    trend.discard((q.tokens-matched)*1000.0/TPS/active);
                    waiting.remove(q.id);c.expired++;if(q.session<0)c.coldExpired++;continue;
                }
                int i=choose(q);if(i<0)continue;
                if(q.escaping)rescued++;
                waiting.remove(q.id);var w=workers[i];q.worker=i;q.reserved=q.tokens-(long)hits[i];
                if(pendingTokens[i]==0 && q.reserved>0)idleCount--;
                w.work.accept(q.reserved);workScores[i]+=q.reserved;pendingTokens[i]+=q.reserved;w.pendingCompute+=q.reserved;w.queue.addLast(q);
                c.routedInput+=q.tokens;c.routedHit+=hits[i];
                if(!w.running && !w.collecting){w.collecting=true;w.collectionDue=now+20;event(w.collectionDue,2,null,i,null);}
                if(!w.running && w.queue.size()>=8)start(i);
            }
            if(now-lastShedding>=100){shed();lastShedding=now;}
        }
        void shed() {
            var candidates=new ArrayList<PrefillQueueShedding.Candidate<Query>>();
            for(Query q:waiting.values()){
                long max=0;
                var cached=cacheIndex.get(q.session);
                Integer sticky=policy==Policy.SESSION && q.migratedFrom<0 ? sessionWorkers.get(q.session) : null;
                if(sticky!=null && sticky>=active)sticky=null;
                if(cached!=null)for(var entry:cached.entrySet()) {
                    Cache cache=entry.getValue();
                    if(entry.getKey()<active && entry.getKey()!=q.migratedFrom
                            && (sticky==null || entry.getKey().equals(sticky)) && now-cache.touched<TTL)
                        max=Math.max(max,Math.min(q.prefix,cache.tokens));
                }
                candidates.add(new PrefillQueueShedding.Candidate<>(q,q.priority,q.id,
                    new PrefillQueueShedding.Estimate(q.tokens,max,Math.round((q.tokens-max)*1000/TPS),active)));
            }
            double queuedWork=candidates.stream().mapToDouble(item->item.estimate().fleetWorkMs()).sum();
            var observed=trend.observe((long)now,queuedWork,(long)SHED_WAIT,(long)MAX_WAIT,
                    candidates.stream().mapToLong(item->item.estimate().workMs()).max().orElse(0));
            if((policy==Policy.ADAPTIVE || policy==Policy.SESSION) && observed.grace())graceTicks++;
            var victims=(policy==Policy.ADAPTIVE || policy==Policy.SESSION) ? PrefillQueueShedding.adaptiveVictims(candidates,sheddingConfig,observed,q -> Math.max(0L,(long)(now-q.at)))
                    : PrefillQueueShedding.victims(candidates,(long)SHED_WAIT);
            for(Query q:victims) {
                if(waiting.remove(q.id)!=null){c.dropped++;droppedPriority[priorityIndex(q)]++;
                    long matched=0;for(int i=0;i<active;i++)matched=Math.max(matched,hit(i,q));
                    trend.discard((q.tokens-matched)*1000.0/TPS/active);if(q.session<0)c.coldDropped++;}
            }
        }
        void start(int i) {
            var w=workers[i];if(w.running)return;
            w.collecting=false;
            var batch=new ArrayList<Query>();long compute=0;
            while(batch.size()<8 && !w.queue.isEmpty()) {
                Query q=w.queue.removeFirst();w.pendingCompute-=q.reserved;
                if(now-q.at>=MAX_WAIT){w.work.release(q.reserved);c.expired++;if(q.session<0)c.coldExpired++;continue;}
                q.hit=hit(i,q);if(policy==Policy.SESSION)sessionWorkers.put(q.session,i);
                q.actualWork=q.tokens-q.hit;
                // A preceding completion or TTL expiry can change actual compute after routing.
                compute+=q.actualWork;batch.add(q);
            }
            if(batch.isEmpty())return;
            w.running=true;w.busyUntil=now+compute*1000/TPS;
            event(w.busyUntil,3,null,i,batch);
        }
        void finish(Event e) {
            var w=workers[e.worker];w.running=false;
            for(Query q:e.batch) {
                w.work.release(q.reserved);
                // Non-preemptible simulated batch consumes its scheduled work, but cannot report
                // success after the request deadline. Failed turns produce neither follow-ups nor cache.
                if(now-q.at>=MAX_WAIT) {
                    c.expired++;if(q.session<0)c.coldExpired++;
                    c.expiredCompute+=q.actualWork;w.compute+=q.actualWork;
                    continue;
                }
                Cache old=w.cache.get(q.session);
                Cache updated=new Cache(Math.max(q.tokens,old==null?0:old.tokens),now);
                w.cache.put(q.session,updated);
                cacheIndex.computeIfAbsent(q.session,unused->new HashMap<>()).put(e.worker,updated);
                w.count++;w.useful+=q.tokens-q.prefix;w.compute+=q.actualWork;w.input+=q.tokens;w.hit+=q.hit;
                c.completed++;completedPriority[priorityIndex(q)]++;if(q.session<0)c.coldCompleted++;c.completedInput+=q.tokens;c.completedHit+=q.hit;c.completedCompute+=q.actualWork;c.completedUseful+=q.tokens-q.prefix;
                latencies.add((now-q.at)/1000);timelineLatencies.add((now-q.at)/1000);waits.add(Math.max(0,(now-q.at-q.actualWork*1000/TPS)/1000));
                if(!independent() && scene!=Scene.REPLAY && q.turn+1<ROUNDS && q.session>0) {
                    double at=now+think(q.session,q.turn+1);
                    event(at,1,new Query(++requestId,q.session,q.turn+1,at,q.base,q.rootAt),-1,null);
                }
            }
            if(!w.queue.isEmpty())start(e.worker);
        }
        long inventory(){return waiting.size()+Arrays.stream(workers).mapToLong(w->w.queue.size()).sum()
            +events.stream().filter(e->e.kind==3).mapToLong(e->e.batch.size()).sum();}
        long[] compute(){return Arrays.stream(workers).mapToLong(w->w.compute).toArray();}
        long[] useful(){return Arrays.stream(workers).mapToLong(w->w.useful).toArray();}
        long[] input(){return Arrays.stream(workers).mapToLong(w->w.input).toArray();}
        void write(PrintWriter out,Counters before,long[] beforeCompute,long[] beforeInput,long[] beforeUseful,double start,double seconds,int stage) {
            long[] compute=compute(),input=input();double[] delta=new double[active],deltaInput=new double[active],deltaUseful=new double[active];
            for(int i=0;i<active;i++){delta[i]=compute[i]-beforeCompute[i];deltaInput[i]=input[i]-beforeInput[i];deltaUseful[i]=workers[i].useful-beforeUseful[i];}
            double offered=c.input-before.input, completed=c.completedInput-before.completedInput;
            double theoretical=c.theoretical-before.theoretical;
            out.printf(Locale.ROOT,"%s,%s,%d,%d,%d,%s,%d,%.3f,%.3f,%.2f,%.3f,%.3f,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%d,%d,%d,%d,%d,%.6f,%.6f,%.3f,%.0f,%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%d,%d%n",
                policy,scene,seed,window,cap,reorder,stage,start,seconds,LOADS[stage],
                (c.arrivals-before.arrivals)/seconds,(c.completed-before.completed)/seconds,
                (offered-theoretical)/seconds/(INITIAL*TPS),cv(delta),cv(deltaInput),maxOverMean(delta),jain(delta),
                offered==0?0:theoretical/offered,
                c.routedInput==before.routedInput?0:(c.routedHit-before.routedHit)/(c.routedInput-before.routedInput),
                completed==0?0:(c.completedHit-before.completedHit)/completed,
                quantile(out==summary?latencies:timelineLatencies,.95),quantile(out==summary?latencies:timelineLatencies,.99),c.dropped-before.dropped,c.expired-before.expired,
                waiting.size(),inventory(),traceHash,cv(deltaUseful),(c.completedCompute-before.completedCompute+c.expiredCompute-before.expiredCompute)/Math.max(1,c.capacityTokens-before.capacityTokens),
                c.arrivals==before.arrivals?0:offered/(c.arrivals-before.arrivals),(c.completedCompute-before.completedCompute)-(c.completedUseful-before.completedUseful),c.coldArrivals-before.coldArrivals,
                c.coldCompleted-before.coldCompleted,c.coldDropped-before.coldDropped,c.coldExpired-before.coldExpired,active,rescued,localMigrations,graceTicks,
                completed/seconds*60,(c.completedCompute-before.completedCompute)/seconds*60,
                quantile(out==summary?latencies:timelineLatencies,.5),c.arrivals-before.arrivals,c.completed-before.completed);
            // Separate columns preserve historical fields while exposing success-cohort TPM and P50.

            out.flush();
            if(out==summary)for(int i=0;i<active;i++)perWorker.printf(Locale.ROOT,
                "%s,%s,%d,%d,%d,%s,%d,%d,%.0f,%.0f,%.3f,%.0f%n",policy,scene,seed,window,cap,reorder,stage,i,delta[i],deltaInput[i],delta[i]/seconds,deltaUseful[i]);
        }
        double burstAt(){return LOADS.length==1 ? endMs/2 : 3.5*stageMs;}
        void resize(int target) {
            if(target<1 || target>N)throw new IllegalArgumentException("target workers");
            if(target<active)for(int i=target;i<active;i++) {
                var worker=workers[i];
                // Graceful drain: already assigned work keeps its owner; new routing excludes it.
                worker.cache.clear();
                final int removed=i;
                cacheIndex.values().forEach(copies->copies.remove(removed));
            }
            active=target;
        }
        void run() {
            event(nextRoot(),0,null,-1,null);event(0,4,null,-1,null);
            for(int s=0;s<LOADS.length;s++){event((s+.5)*stageMs,6,null,s,null);event((s+1)*stageMs,7,null,s,null);}
            for(double at=30_000;at<=endMs;at+=30_000)event(at,8,null,-1,null);
            if(scene==Scene.COLD_BURST || scene==Scene.COLD_BURST_OPEN)event(burstAt(),5,null,-1,null);
            if(scene==Scene.SCALE_OUT || scene==Scene.HOTSPOT_SCALE_OUT)event(endMs*Double.parseDouble(System.getProperty("flexlb.sim.resizeAt", ".75")),9,null,-1,null);
            if(scene==Scene.SCALE_IN)event(endMs*.5,11,null,-1,null);
            if(scene==Scene.SCALE_CYCLE){event(endMs*.33,9,null,-1,null);event(endMs*.66,11,null,-1,null);}
            if(scene==Scene.RECOVERY_BURST)event(endMs/2,10,null,-1,null);
            while(!events.isEmpty() && events.peek().at<=endMs) {
                Event e=events.poll();
                int serving=active;
                for(int i=active;i<N;i++)if(workers[i].running || !workers[i].queue.isEmpty())serving++;
                c.capacityTokens+=(e.at-now)*serving*TPS/1000;now=e.at;
                switch(e.kind) {
                    case 9 -> resize(Integer.getInteger("flexlb.sim.scaleOut", N));
                    case 11 -> resize(Integer.getInteger("flexlb.sim.scaleIn", INITIAL/2));
                    case 10 -> {
                        for(int i=0;i<3000;i++)arrive(new Query(++requestId,-requestId,0,now,base()*MEAN_FACTOR,now));
                    }
                    case 0 -> {
                        long session=++sessionId;double base=base();
                        if(independent()){session=1+lengths.nextInt(corpus.length);base=corpus[(int)session-1];}
                        traceHash=31*traceHash+Double.doubleToLongBits(base);
                        Query q=new Query(++requestId,session,0,now,base,now);
                        if(independent()){q.tokens=Math.round(base);q.prefix=Math.round(HIT*q.tokens);}
                        arrive(q);
                        if(scene==Scene.REPLAY){double at=now;for(int t=1;t<ROUNDS;t++){
                            at+=think(session,t);event(at,1,new Query(++requestId,session,t,at,base,now),-1,null);}}
                        event(nextRoot(),0,null,-1,null);
                    }
                    case 1 -> arrive(e.query);
                    case 2 -> {if(workers[e.worker].collecting && workers[e.worker].collectionDue==e.at)start(e.worker);}
                    case 3 -> finish(e);
                    case 4 -> {route();event(now+20,4,null,-1,null);}
                    case 5 -> {
                        // 70 cold QPS * 100k = one entire fleet's extra compute, for 60 simulated seconds.
                        Query q=new Query(++requestId,-requestId,0,now,base()*MEAN_FACTOR,now);arrive(q);
                        if(now<burstAt()+60_000)event(now+1000.0/70,5,null,-1,null);
                    }
                    case 6 -> {measured=c.copy();measuredCompute=compute();measuredInput=input();measuredUseful=useful();latencies.clear();waits.clear();}
                    case 7 -> {write(summary,measured,measuredCompute,measuredInput,measuredUseful,(e.worker+.5)*stageMs/1000,stageMs/2000,e.worker);}
                    case 8 -> {
                        int stage=Math.min(LOADS.length-1,(int)((now-1)/stageMs));
                        write(timeline,prev,prevCompute,prevInput,prevUseful,now/1000-30,30,stage);
                        prev=c.copy();prevCompute=compute();prevInput=input();prevUseful=useful();timelineLatencies.clear();
                        // Remove stale unused sessions too, keeping memory bounded beyond TTL.
                        for(var w:workers)w.cache.entrySet().removeIf(entry->now-entry.getValue().touched>=TTL);
                        cacheIndex.values().forEach(copies->copies.values().removeIf(cache->now-cache.touched>=TTL));
                        cacheIndex.values().removeIf(Map::isEmpty);
                    }
                }
            }
            if(c.arrivals!=c.completed+c.dropped+c.expired+inventory())throw new AssertionError("request conservation");
            double workerCompute=Arrays.stream(workers).mapToDouble(w->w.compute).sum();
            if(workerCompute!=c.completedCompute+c.expiredCompute || c.completedInput!=c.completedHit+c.completedCompute)throw new AssertionError("token conservation");
            if(c.completedCompute+c.expiredCompute>c.capacityTokens+1)throw new AssertionError("capacity exceeded");
            for(int i=0;i<N;i++) {
                Worker w=workers[i];long reserved=w.queue.stream().mapToLong(q->q.reserved).sum();
                for(Event event:events)if(event.kind==3 && event.worker==i)reserved+=event.batch.stream().mapToLong(q->q.reserved).sum();
                if(w.work.snapshot().pendingTokens()!=reserved)throw new AssertionError("reservation conservation");
            }
            long newWorkerCompute=0;for(int i=INITIAL;i<N;i++)newWorkerCompute+=workers[i].compute;
            System.out.printf("NEW_WORKER_COMPUTE policy=%s scene=%s seed=%d tokens=%d%n",policy,scene,seed,newWorkerCompute);
            System.out.printf("ELASTIC policy=%s scene=%s seed=%d active=%d rescued=%d local_migrations=%d grace_ticks=%d drops_priority=%s success_priority=%s%n",
                    policy,scene,seed,active,rescued,localMigrations,graceTicks,Arrays.toString(droppedPriority),Arrays.toString(completedPriority));
            System.out.printf("PASS %s %s seed=%d window=%d cap=%d reorder=%s arrivals=%d success=%d drops=%d expire=%d inventory=%d%n",
                policy,scene,seed,window,cap,reorder,c.arrivals,c.completed,c.dropped,c.expired,inventory());
        }
    }
    static double growth(){double lo=1,hi=1.2;for(int n=0;n<100;n++){double r=(lo+hi)/2,s=0;for(int i=0;i<ROUNDS;i++)s+=Math.pow(r,i);
        if(Math.pow(r,ROUNDS-1)/s>.1)hi=r;else lo=r;}return (lo+hi)/2;}
    static double sumGrowth(){double s=0;for(int i=0;i<ROUNDS;i++)s+=Math.pow(GROWTH,i);return s;}
    static double cv(double[] a){double mean=Arrays.stream(a).average().orElse(0),v=0;for(double x:a)v+=(x-mean)*(x-mean);return mean==0?0:Math.sqrt(v/a.length)/mean;}
    static double maxOverMean(double[] a){double mean=Arrays.stream(a).average().orElse(0);return mean==0?0:Arrays.stream(a).max().orElse(0)/mean;}
    static double jain(double[] a){double sum=Arrays.stream(a).sum(),sq=Arrays.stream(a).map(x->x*x).sum();return sq==0?1:sum*sum/(a.length*sq);}
    static double quantile(List<Double> values,double p){if(values.isEmpty())return 0;var sorted=new ArrayList<>(values);Collections.sort(sorted);return sorted.get(Math.min(sorted.size()-1,(int)(p*sorted.size())));}
    public static void main(String[] args)throws Exception {
        Path out=Path.of(args[0]);Files.createDirectories(out);
        int seeds=args.length>1?Integer.parseInt(args[1]):3;
        double seconds=args.length>2?Double.parseDouble(args[2]):300;
        long window=args.length>3?Long.parseLong(args[3]):10000,cap=args.length>4?Long.parseLong(args[4]):140000;
        boolean reorder=args.length>5&&Boolean.parseBoolean(args[5]);
        String scenes=args.length>6?args[6]:"REPLAY,CONVERSATION,COLD_BURST,TTL_GAP";
        try(var summary=new PrintWriter(Files.newBufferedWriter(out.resolve("summary.csv")));
            var timeline=new PrintWriter(Files.newBufferedWriter(out.resolve("timeline.csv")));
            var workers=new PrintWriter(Files.newBufferedWriter(out.resolve("workers.csv")))) {
            String header="policy,scene,seed,window_ms,cap,reorder,stage,start_s,duration_s,nominal_load,arrival_qps,success_qps,offered_compute_load,compute_cv,input_cv,max_mean,jain,theoretical_hit,routed_hit,success_hit,p95_s,p99_s,dropped,expired,global_queue,inventory,trace_hash,useful_compute_cv,actual_compute_load,avg_input_tokens,extra_compute_tokens,cold_arrivals,cold_completed,cold_dropped,cold_expired,active_workers,cumulative_rescues,cumulative_local_migrations,cumulative_grace_ticks,input_tpm,compute_tpm,p50_s,arrivals,completed";
            summary.println(header);timeline.println(header);
            workers.println("policy,scene,seed,window_ms,cap,reorder,stage,worker,compute_tokens,input_tokens,compute_tps,useful_compute_tokens");
            for(String scene:scenes.split(","))for(int seed=1;seed<=seeds;seed++)for(Policy p:Arrays.stream(System.getProperty("flexlb.sim.policies", "SHORT_TTFT,SHORT_TTFT_CAPPED,CACHE_ONLY,CACHE_WINDOW").split(",")).map(Policy::valueOf).toList())
                new Run(p,Scene.valueOf(scene),seed,seconds,window,cap,reorder,summary,timeline,workers).run();
        }
    }
}
