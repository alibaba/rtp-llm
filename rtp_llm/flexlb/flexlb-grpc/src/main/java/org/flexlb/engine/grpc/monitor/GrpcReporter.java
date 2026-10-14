package org.flexlb.engine.grpc.monitor;

import com.google.protobuf.MessageLite;
import io.grpc.Status;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import javax.annotation.PostConstruct;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.DoubleConsumer;
import static org.flexlb.constant.MetricConstant.*;

/** Complete logical RPC observations; protobuf bytes are not wire or heap bytes. */
@Component
public class GrpcReporter {
    private static final Logger LOG = LoggerFactory.getLogger("syncLogger");
    private static final int MAX_BINDINGS = 4096;
    private final FlexMonitor monitor;
    private final ConcurrentHashMap<String,String> roles=new ConcurrentHashMap<>();
    public void registerRole(String ip,int port,String role){
        String address=ip+":"+port;if(roles.size()>=MAX_BINDINGS&&!roles.containsKey(address))return;
        String old=roles.put(address,role);if(!java.util.Objects.equals(old,role))peers.entrySet().removeIf(e->e.getValue().ip.equals(ip)&&e.getValue().port.equals(Integer.toString(port)));
    }
    private final ConcurrentHashMap<String, Peer> peers = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Peer> overflow = new ConcurrentHashMap<>();
    private DoubleConsumer overflows = value -> { };
    public GrpcReporter(FlexMonitor monitor) { this.monitor = monitor; }
    @PostConstruct public void init() {
        monitor.register(GRPC_CHANNEL_POOL_SIZE, FlexMetricType.GAUGE);
        monitor.register(GRPC_CALL_DURATION, FlexMetricType.TIMER);
        monitor.register(GRPC_RESPONSE_SIZE, FlexMetricType.DISTRIBUTION);
        monitor.register(GRPC_CALL_COUNT, FlexMetricType.QPS);
        monitor.register(GRPC_CONNECTION_DURATION, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register("app.grpc.metric.binding.overflow", FlexMetricType.COUNTER);
        overflows=monitor.bind("app.grpc.metric.binding.overflow",FlexMetricTags.of());
    }
    public void reportChannelPoolSize(int size) { monitor.report(GRPC_CHANNEL_POOL_SIZE, FlexMetricTags.of("type","grpc"), size); }
    public void reportConnectionDuration(String service, long duration) { monitor.report(GRPC_CONNECTION_DURATION,FlexMetricTags.of("service",service),duration); }
    public Call start(String key, String ip, int port, String service, long startNs) {
        try { return startBound(key,ip,port,service,startNs); }
        catch(RuntimeException ignored) { return null; }
    }
    private Call startBound(String key,String ip,int port,String service,long startNs) {
        Peer peer=peers.get(key);
        if(peer==null){
            synchronized(peers){
                peer=peers.get(key);
                if(peer==null){
                    if(peers.size()<MAX_BINDINGS){peer=new Peer(ip,Integer.toString(port),service);peers.put(key,peer);}
                    else{peer=overflow.computeIfAbsent(service,s->new Peer("overflow","overflow",s));overflows.accept(1);}
                }
            }
        }
        return new Call(peer,startNs);
    }
    private final class Peer {
        final String ip, port, service, role;
        final DoubleConsumer[] duration=new DoubleConsumer[2],bytes=new DoubleConsumer[2];
        final DoubleConsumer[][] results=new DoubleConsumer[2][Status.Code.values().length];
        final AtomicLong lastErrorLogNs=new AtomicLong(),suppressed=new AtomicLong();
        Peer(String ip,String port,String service){this.ip=ip;this.port=port;this.service=service;this.role=roles.getOrDefault(ip+":"+port,"unknown");
            for(int r=0;r<2;r++){
                FlexMetricTags tags=tags(r,null);
                duration[r]=monitor.bind(GRPC_CALL_DURATION,tags);
                bytes[r]=monitor.bind(GRPC_RESPONSE_SIZE,tags);
            }
        }
        FlexMetricTags tags(int retry,String code){return code==null
                ?FlexMetricTags.of("engineIp",ip,"enginePort",port,"service",service,"role",role,"retry",Boolean.toString(retry==1))
                :FlexMetricTags.of("engineIp",ip,"enginePort",port,"service",service,"role",role,"retry",Boolean.toString(retry==1),"code",code);}
        DoubleConsumer result(int retry,Status.Code code){
            DoubleConsumer c=results[retry][code.ordinal()];
            if(c==null){synchronized(results){c=results[retry][code.ordinal()];if(c==null){c=monitor.bind(GRPC_CALL_COUNT,tags(retry,code.name()));results[retry][code.ordinal()]=c;}}}
            return c;
        }
    }
    public final class Call {
        private final Peer peer;
        private final long startNs;
        private boolean finished;
        private volatile boolean retried;
        Call(Peer peer,long startNs){this.peer=peer;this.startNs=startNs;}
        public void retryStarted(){retried=true;}
        public void complete(Object response,Throwable failure){
            if(finished){return;}finished=true;
            int retry=retried?1:0;Status.Code code=failure==null?Status.Code.OK:Status.fromThrowable(failure).getCode();
            double ms=(System.nanoTime()-startNs)/1e6;
            try{
                peer.duration[retry].accept(ms);peer.result(retry,code).accept(1);
                if(failure==null&&response instanceof MessageLite pb){peer.bytes[retry].accept(pb.getSerializedSize());}
                if(failure!=null){
                    long now=System.nanoTime(),last=peer.lastErrorLogNs.get();
                    if(now-last>=10_000_000_000L&&peer.lastErrorLogNs.compareAndSet(last,now)){
                        LOG.warn("event=rpc_failed engineIp={} enginePort={} service={} retry={} code={} duration_ms={} cause={} suppressed={}",peer.ip,peer.port,peer.service,retried,code,ms,failure.getClass().getSimpleName(),peer.suppressed.getAndSet(0));
                    }else{peer.suppressed.incrementAndGet();}
                }
            }catch(RuntimeException ignored){ /* Monitoring cannot fail the RPC. */ }
        }
    }
}
