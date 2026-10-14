package org.flexlb.service.monitor;

import com.sun.management.GarbageCollectionNotificationInfo;
import org.flexlb.cache.core.EngineLocalView;
import org.flexlb.cache.core.GlobalCacheIndex;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.diagnostics.ObservedExecutor;
import org.flexlb.diagnostics.SyncObservation;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.sync.status.WorkerDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import javax.management.*;
import javax.management.openmbean.CompositeData;
import java.lang.management.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.DoubleConsumer;

/** One-second O(workers + executors) sampling. Never walks key sets or retains payloads. */
@Component
public final class SynchronizationMetrics {
    private static final Logger LOG=LoggerFactory.getLogger("syncLogger");
    private static final int MAX_WORKERS=4096, GC_BUFFER_SIZE=128;
    private final FlexMonitor monitor;
    private final org.flexlb.engine.grpc.EngineGrpcClient grpcClient;
    private final WorkerDirectory directory;
    private final EngineLocalView local;
    private final GlobalCacheIndex global;
    private final Map<String,WorkerMeters> workers=new ConcurrentHashMap<>();
    private final Map<String,PoolMeters> pools=new HashMap<>();
    private final Map<String,DoubleConsumer> gcTimers=new HashMap<>();
    private final List<NotificationEmitter> emitters=new ArrayList<>();
    private final List<BufferPoolMXBean> buffers=ManagementFactory.getPlatformMXBeans(BufferPoolMXBean.class);
    private final ScheduledExecutorService sampler=Executors.newSingleThreadScheduledExecutor(r->{Thread t=new Thread(r,"sync-metrics-sampler");t.setDaemon(true);return t;});
    private final NotificationListener listener=this::gc;
    private final GcEvent[] gcRing=new GcEvent[GC_BUFFER_SIZE];
    private int gcWrite,gcRead,gcCount;
    private long droppedGc;
    private volatile long sampleUs,cacheEntries,statusTasks,localEntries,globalEntries,globalMappings,pendingPayloadBytes;
    private record StateBudget(long epochMs,long cache,long tasks,long local,long global,long mappings,long pending) { }
    private volatile StateBudget budget = new StateBudget(0,0,0,0,0,0,0);
    private final GcEvent gcScratch = new GcEvent();
    private final java.util.Set<String> heapPoolNames = new java.util.HashSet<>();
    private DoubleConsumer heapUsed,heapMax,directUsed,objectCache,objectStatus,objectLocal,objectGlobal,pendingBytes,gcDrops,cacheCount,statusCount,localCount,globalCount,mappingCount,bindingDrops;
    private final java.util.concurrent.atomic.AtomicLong droppedBindings = new java.util.concurrent.atomic.AtomicLong();
    private static final class GcEvent {long id,startEpoch,endEpoch,duration,before,after;String collector,cause;StateBudget budget;}

    public SynchronizationMetrics(FlexMonitor monitor,WorkerDirectory directory,EngineLocalView local,GlobalCacheIndex global,org.flexlb.engine.grpc.EngineGrpcClient grpcClient){
        this.grpcClient=grpcClient;this.monitor=monitor;this.directory=directory;this.local=local;this.global=global;
        for(int i=0;i<GC_BUFFER_SIZE;i++)gcRing[i]=new GcEvent();
        for(MemoryPoolMXBean pool:ManagementFactory.getMemoryPoolMXBeans())if(pool.getType()==MemoryType.HEAP)heapPoolNames.add(pool.getName());
    }
    @PostConstruct public void start(){
        for(String name:List.of("app.sync.phase","app.sync.phase.age.ms","app.sync.response.age.ms","app.sync.commit.age.ms","app.sync.pending.response.bytes","app.executor.oldest.pending.ms","app.executor.oldest.running.ms","app.executor.queue.capacity","app.executor.rejected.total","app.executor.caller.runs.total","app.engine.balancing.thread.pool.info","app.executor.completed.tasks","app.netty.event.loop.lag.ms","app.jvm.heap.used.bytes","app.jvm.heap.max.bytes","app.jvm.direct.used.bytes","app.sync.object.entry.budget.bytes","app.sync.object.entries","app.jvm.gc.events.dropped","app.sync.metric.binding.overflow"))monitor.register(name,FlexMetricType.GAUGE);
        for(String name:List.of("app.sync.convert.ms","app.sync.callback.queue.ms","app.sync.apply.ms","app.jvm.gc.pause.ms"))monitor.register(name,FlexMetricType.TIMER);
        heapUsed=bind("app.jvm.heap.used.bytes");heapMax=bind("app.jvm.heap.max.bytes");directUsed=bind("app.jvm.direct.used.bytes");pendingBytes=bind("app.sync.pending.response.bytes");gcDrops=bind("app.jvm.gc.events.dropped");
        cacheCount=objectCount("cache_snapshot");statusCount=objectCount("worker_status_tasks");localCount=objectCount("local_index");globalCount=objectCount("global_index_keys");mappingCount=objectCount("global_index_mappings");bindingDrops=bind("app.sync.metric.binding.overflow");
        objectCache=objectBudget("cache_snapshot");objectStatus=objectBudget("worker_status_tasks");objectLocal=objectBudget("local_index");objectGlobal=objectBudget("global_index");
        for(GarbageCollectorMXBean gc:ManagementFactory.getGarbageCollectorMXBeans())if(gc instanceof NotificationEmitter emitter){emitter.addNotificationListener(listener,null,null);emitters.add(emitter);}
        sampler.scheduleWithFixedDelay(()->{try{sample();}catch(RuntimeException e){LOG.warn("event=metrics_sample_failed cause={}",e.getClass().getSimpleName());}},0,1,TimeUnit.SECONDS);
    }
    private DoubleConsumer bind(String name){return monitor.bind(name,FlexMetricTags.of());}
    private DoubleConsumer objectBudget(String kind){return monitor.bind("app.sync.object.entry.budget.bytes",FlexMetricTags.of("object_kind",kind,"estimate_method","cardinality_budget_v1"));}
    private DoubleConsumer objectCount(String kind){return monitor.bind("app.sync.object.entries",FlexMetricTags.of("object_kind",kind));}
    public void ensure(WorkerStatus status){
        if(status.observationBound)return;
        synchronized(status.statusObservation){
            if(status.observationBound)return;
            grpcClient.registerPeerRole(status.getIp(),status.getGrpcPort(),status.getRole().name());
            String address=status.getIpPort()+"|"+status.getRole();WorkerMeters meters=workers.get(address);
            if(meters==null){synchronized(workers){meters=workers.get(address);if(meters==null&&workers.size()<MAX_WORKERS){meters=new WorkerMeters(status);workers.put(address,meters);}}}
            if(meters!=null){meters.attach(status);}else{droppedBindings.incrementAndGet();}
            status.observationBound=true;
        }
    }
    private final class WorkerMeters {
        long seenCycle;
        final DoubleConsumer[][] value=new DoubleConsumer[2][4],timing=new DoubleConsumer[2][3];
        WorkerMeters(WorkerStatus status){
            for(int k=0;k<2;k++){
                FlexMetricTags tags=FlexMetricTags.of("engineIp",status.getIp(),"enginePort",Integer.toString(status.getGrpcPort()),"role",status.getRole().name(),"service",k==0?"status":"cache");
                String[] names={"app.sync.phase","app.sync.phase.age.ms","app.sync.response.age.ms","app.sync.commit.age.ms"};
                for(int i=0;i<4;i++)value[k][i]=monitor.bind(names[i],tags);
                String[] names2={"app.sync.convert.ms","app.sync.callback.queue.ms","app.sync.apply.ms"};
                for(int i=0;i<3;i++)timing[k][i]=monitor.bind(names2[i],tags);
            }
        }
        void attach(WorkerStatus status){for(int k=0;k<2;k++){SyncObservation s=k==0?status.statusObservation:status.cacheObservation;s.conversion=timing[k][0];s.queue=timing[k][1];s.apply=timing[k][2];}}
        void sample(WorkerStatus status,long now,long cycle){seenCycle=cycle;
            for(int k=0;k<2;k++){SyncObservation s=k==0?status.statusObservation:status.cacheObservation;
                value[k][0].accept(s.phase);value[k][1].accept(s.phaseAgeUs(now)/1000.0);
                value[k][2].accept(s.lastAliveResponseUs>0?(now-s.lastAliveResponseUs)/1000.0:Double.NaN);
                long commit=k==0?status.pollHealth().lastSuccessfulPollUs():status.lastSuccessfulCachePollUs();
                boolean valid=k==0?status.appliedStatusCursor().statusVersion()>0:commit>0;
                value[k][3].accept(valid?(now-commit)/1000.0:Double.NaN);
            }
        }
        void absent(){for(int k=0;k<2;k++){value[k][0].accept(0);value[k][1].accept(0);value[k][2].accept(Double.NaN);value[k][3].accept(Double.NaN);}}
    }
    private final class PoolMeters {
        final DoubleConsumer active,queued,size,completed,capacity,waiting,running,rejected,caller;
        PoolMeters(String name){FlexMetricTags tags=FlexMetricTags.of("threadPool",name);
            active=monitor.bind("app.engine.balancing.thread.pool.info",FlexMetricTags.of("threadPool",name,"type","executingTaskThreadSize"));
            queued=monitor.bind("app.engine.balancing.thread.pool.info",FlexMetricTags.of("threadPool",name,"type","queueSize"));
            size=monitor.bind("app.engine.balancing.thread.pool.info",FlexMetricTags.of("threadPool",name,"type","currentThreadSizeInPool"));
            completed=monitor.bind("app.executor.completed.tasks",tags);
            capacity=monitor.bind("app.executor.queue.capacity",tags);waiting=monitor.bind("app.executor.oldest.pending.ms",tags);running=monitor.bind("app.executor.oldest.running.ms",tags);rejected=monitor.bind("app.executor.rejected.total",tags);caller=monitor.bind("app.executor.caller.runs.total",tags);
        }
        void sample(ObservedExecutor pool){pool.bind(monitor);active.accept(pool.getActiveCount());queued.accept(pool.getQueue().size());size.accept(pool.getPoolSize());completed.accept(pool.getCompletedTaskCount());capacity.accept(pool.getQueue().size()+pool.getQueue().remainingCapacity());waiting.accept(pool.oldestPendingMs());running.accept(pool.oldestRunningMs());rejected.accept(pool.rejectionCount());caller.accept(pool.callerRunCount());}
    }
    private void sample(){
        long now=System.nanoTime()/1000,cache=0,tasks=0,locals=0,pending=0;
        for(RoleType role:RoleType.values())for(WorkerStatus status:directory.statusSnapshot(role).values()){
            ensure(status);WorkerMeters m=workers.get(status.getIpPort()+"|"+status.getRole());if(m!=null)m.sample(status,now,now);
            var snapshot=status.getCacheStatus();if(snapshot!=null)cache+=snapshot.getCacheKeySize();
            tasks+=status.committedEngineObservation().runningTaskList().size();
            if(status.statusObservation.phase>0)tasks+=status.statusObservation.finishedTaskCount;
            locals+=local.size(status.getIpPort());pending+=status.statusObservation.payloadBytes+status.cacheObservation.payloadBytes;
        }
        for(WorkerMeters m:workers.values())if(m.seenCycle!=now)m.absent();
        cacheEntries=cache;statusTasks=tasks;localEntries=locals;globalEntries=global.totalBlocks();globalMappings=global.totalMappings();pendingPayloadBytes=pending;sampleUs=now;
        budget=new StateBudget(System.currentTimeMillis(),cache,tasks,locals,globalEntries,globalMappings,pending);
        // Entry budgets exclude variable strings, sharing, temporary diff sets and VM layout details.
        cacheCount.accept(cache);statusCount.accept(tasks);localCount.accept(locals);globalCount.accept(globalEntries);mappingCount.accept(globalMappings);bindingDrops.accept(droppedBindings.get());
        objectCache.accept(cache*64.0);objectStatus.accept(tasks*256.0);objectLocal.accept(locals*64.0);objectGlobal.accept(globalEntries*64.0+globalMappings*48.0);pendingBytes.accept(pending);
        MemoryUsage heap=ManagementFactory.getMemoryMXBean().getHeapMemoryUsage();heapUsed.accept(heap.getUsed());heapMax.accept(heap.getMax());
        long direct=0;for(BufferPoolMXBean b:buffers)if(b.getName().equals("direct"))direct+=b.getMemoryUsed();directUsed.accept(direct);
        ObservedExecutor.forEachLive(p->pools.computeIfAbsent(p.metricName(),PoolMeters::new).sample(p));
        drainGc();
    }
    private synchronized void gc(Notification notification,Object ignored){
        if(!notification.getType().equals(GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION))return;
        GarbageCollectionNotificationInfo info=GarbageCollectionNotificationInfo.from((CompositeData)notification.getUserData());
        if(gcCount==GC_BUFFER_SIZE){droppedGc++;return;}
        GcEvent e=gcRing[gcWrite];var g=info.getGcInfo();long boot=ManagementFactory.getRuntimeMXBean().getStartTime();
        e.id=g.getId();e.startEpoch=boot+g.getStartTime();e.endEpoch=boot+g.getEndTime();e.duration=g.getDuration();e.collector=info.getGcName();e.cause=info.getGcCause();e.before=0;e.after=0;
        for(var entry:g.getMemoryUsageBeforeGc().entrySet())if(heapPoolNames.contains(entry.getKey()))e.before+=entry.getValue().getUsed();
        for(var entry:g.getMemoryUsageAfterGc().entrySet())if(heapPoolNames.contains(entry.getKey()))e.after+=entry.getValue().getUsed();
        e.budget=budget;
        gcWrite=(gcWrite+1)%GC_BUFFER_SIZE;gcCount++;
    }
    private synchronized boolean takeGc(){
        if(gcCount==0)return false;
        GcEvent source=gcRing[gcRead],e=gcScratch;
        e.id=source.id;e.startEpoch=source.startEpoch;e.endEpoch=source.endEpoch;e.duration=source.duration;e.before=source.before;e.after=source.after;e.collector=source.collector;e.cause=source.cause;e.budget=source.budget;
        source.budget=null;gcRead=(gcRead+1)%GC_BUFFER_SIZE;gcCount--;return true;
    }
    private void drainGc(){
        while(takeGc()){
            GcEvent e=gcScratch;StateBudget b=e.budget;
            gcTimers.computeIfAbsent(e.collector,name->monitor.bind("app.jvm.gc.pause.ms",FlexMetricTags.of("collector",name))).accept(e.duration);
            LOG.info("event=gc_state gc_id={} collector={} cause={} start_epoch_ms={} end_epoch_ms={} pause_ms={} heap_before_bytes={} heap_after_bytes={} state_sample_epoch_ms={} sample_age_at_gc_end_ms={} cache_entries={} status_tasks={} local_entries={} global_entries={} global_mappings={} pending_response_bytes={} estimate_method=cardinality_budget_v1 cache_entry_budget_bytes={} status_entry_budget_bytes={} local_entry_budget_bytes={} global_entry_budget_bytes={}",
                    e.id,e.collector,e.cause,e.startEpoch,e.endEpoch,e.duration,e.before,e.after,b.epochMs(),b.epochMs()>0?e.endEpoch-b.epochMs():-1,b.cache(),b.tasks(),b.local(),b.global(),b.mappings(),b.pending(),b.cache()*64,b.tasks()*256,b.local()*64,b.global()*64+b.mappings()*48);
        }
        synchronized(this){gcDrops.accept(droppedGc);}
    }
    @PreDestroy public void close(){sampler.shutdownNow();for(NotificationEmitter emitter:emitters)try{emitter.removeNotificationListener(listener);}catch(ListenerNotFoundException ignored){}}
}
