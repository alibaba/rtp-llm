import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import org.flexlb.balance.delivery.*;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.*;
import org.flexlb.balance.projection.*;
import org.flexlb.balance.scheduler.*;
import static org.mockito.Mockito.mock;

/** Contended real projection calls; all revisions run this identical source. */
public class ConcurrentBench {
    static final ThreadMXBean MX = (ThreadMXBean) ManagementFactory.getThreadMXBean();
    static volatile long stopAt;
    static volatile long sink;
    record Stats(long count, long cpu, long bytes, long blockedMs, long blockedCount, long[] histogram) { }
    static RouteProjection.Inputs fresh(RouteProjection.Inputs input) {
        return new RouteProjection.Inputs(input.queue(), input.work(), input.ownershipVersion() + 1);
    }
    static void bench(FormulaPredictor model, RouteProjection.DeliveryProjection policy,
                      String name, int threads, int endpoints, int replaceEvery) throws Exception {
        var refs = new AtomicReferenceArray<RouteProjection.Inputs>(endpoints);
        for (int e = 0; e < endpoints; e++) {
            var items = new ArrayList<GroupPlanner.Item>();
            for (int i = 0; i < 128; i++) items.add(new GroupPlanner.Item(i, 0, i, 1000,
                    1_000_000, 1000 + i + e, i % 100));
            refs.set(e, new RouteProjection.Inputs(new QueueSnapshot(1000, true,
                    Comparator.comparingLong(GroupPlanner.Item::enqueueSeq),
                    new GroupPlanner.Constraints(64, 10_000_000, 20_000_000, 700, 0), items, null),
                    new WorkSnapshot(1000, List.of(), List.of(), 0)));
        }
        var probes = new RouteProjection.Probe[64];
        var expected = new RouteProjection.Candidate[64][endpoints];
        for (int p = 0; p < 64; p++) {
            probes[p] = new RouteProjection.Probe(999999, 0, 1000, 1_000_000,
                    1024 + p * 32, p * 8, p * 8);
            for (int e = 0; e < endpoints; e++) expected[p][e] = RouteProjection.project(refs.get(e), probes[p], model, policy);
        }
        var sequence = new AtomicLong();
        try (var pool = Executors.newFixedThreadPool(threads)) {
            for (int round = 0; round < 3; round++) {
                var barrier = new CyclicBarrier(threads + 1);
                var futures = new ArrayList<Future<Stats>>();
                for (int t = 0; t < threads; t++) {
                    int thread = t;
                    futures.add(pool.submit(() -> {
                        barrier.await();
                        long id = Thread.currentThread().threadId();
                        var info = MX.getThreadInfo(id);
                        long blocked = info.getBlockedTime(), blocks = info.getBlockedCount();
                        long cpu = MX.getCurrentThreadCpuTime(), bytes = MX.getThreadAllocatedBytes(id);
                        long count = 0, checksum = 0;
                        long[] histogram = new long[2048];
                        while (System.nanoTime() < stopAt) {
                            long start = System.nanoTime();
                            int p = (int)(count + thread) & 63;
                            long seq = sequence.getAndIncrement();
                            if (replaceEvery > 0 && seq % replaceEvery == 0) {
                                // A whole fresh endpoint snapshot is atomically published.
                                int endpoint = (int)(seq / replaceEvery % endpoints);
                                refs.updateAndGet(endpoint, ConcurrentBench::fresh);
                            }
                            for (int e = 0; e < endpoints; e++) {
                                var result = RouteProjection.project(refs.get(e), probes[p], model, policy);
                                if (!expected[p][e].equals(result)) throw new AssertionError("projection mismatch");
                                checksum += result.requiredProjectedTtftMs();
                            }
                            long elapsed = System.nanoTime() - start;
                            int bucket = Math.min(2047, (int)(Math.log1p(elapsed) / Math.log(1.01)));
                            histogram[bucket]++;
                            count++;
                        }
                        long used = MX.getCurrentThreadCpuTime() - cpu;
                        long allocated = MX.getThreadAllocatedBytes(id) - bytes;
                        info = MX.getThreadInfo(id);
                        sink = checksum;
                        return new Stats(count, used, allocated, info.getBlockedTime()-blocked,
                                info.getBlockedCount()-blocks, histogram);
                    }));
                }
                long start = System.nanoTime();
                stopAt = start + 700_000_000L;
                barrier.await();
                long count=0,cpu=0,bytes=0,blocked=0,blocks=0;
                long[] hist = new long[2048];
                for (var future : futures) {
                    var s=future.get();count+=s.count;cpu+=s.cpu;bytes+=s.bytes;blocked+=s.blockedMs;blocks+=s.blockedCount;
                    for(int i=0;i<hist.length;i++)hist[i]+=s.histogram[i];
                }
                long wall=System.nanoTime()-start;
                if(round>0)System.out.printf(Locale.ROOT,
                        "%s t=%d qps=%.2f cpu_ns=%.2f bytes=%.2f p99_us=%.2f blocked_us=%.2f blocks=%.4f%n",
                        name,threads,count*1e9/wall,(double)cpu/count,(double)bytes/count,
                        percentile(hist,count,.99)/1000.,blocked*1000./count,(double)blocks/count);
            }
        }
    }
    static double percentile(long[] hist,long count,double fraction) {
        long n=0;for(int i=0;i<hist.length;i++){n+=hist[i];if(n>=count*fraction)return Math.expm1((i+1)*Math.log(1.01));}return 0;
    }
    public static void main(String[] args) throws Exception {
        MX.setThreadContentionMonitoringEnabled(true);
        var model=new FormulaPredictor(Files.readString(Path.of(args[0])));
        var registry=mock(RequestRegistry.class);var metrics=mock(DeliveryMetrics.class);
        var batch=new BatchDeliveryStrategy(()->CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST),
                ()->1L,registry,metrics).projectionPolicy();
        if(args.length>1) {
            bench(model,batch,"batch_shared_profile",64,1,0);
            return;
        }
        for(int t:new int[]{1,8,64}) {
            bench(model,batch,"batch_shared",t,1,0);
            bench(model,batch,"batch_fleet10",t,10,0);
            bench(model,batch,"batch_churn",t,1,1);
        }
        bench(model,new RouteDeliveryStrategy(registry,metrics).projectionPolicy(),"route_shared",64,1,0);
    }
}
