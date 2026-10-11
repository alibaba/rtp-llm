package org.flexlb.balance.scheduler;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.classic.spi.ThrowableProxy;
import ch.qos.logback.core.read.ListAppender;
import org.slf4j.LoggerFactory;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;
class RequestDeliveryPreparationLoggingTest {
 @Test void preservesOriginalFailureAndRequestContextWithoutDuplicateStaleLog() throws Exception {
  Logger logger=(Logger)LoggerFactory.getLogger("flexlbLogger");
  Level previous=logger.getLevel();
  ListAppender<ILoggingEvent> sink=new ListAppender<>(); sink.start(); logger.addAppender(sink); logger.setLevel(Level.ERROR);
  var config=SchedulingTestConfig.batchConfig();
  ConfigService service=mock(ConfigService.class); when(service.loadBalanceConfig()).thenReturn(config);
  var registry=new RequestRegistry(service,mock(BatchSchedulerReporter.class),mock(RequestSchedulerReporter.class),mock(org.flexlb.balance.eviction.EngineCancelChannel.class));
  try {
   var context=RequestLifecycleTestSupport.context(config,903L);
   var future=registry.register(context, 100); var slot=registry.requestSlot(903L);
   var item=new ScheduledRequest(context,future,new Response(),null,null,mock(PrefillEndpoint.class),null,null,slot.createdAtMs());
   var failure=new IllegalStateException("preparation cause",new IllegalArgumentException("nested cause"));
   try(var admission=registry.claimAdmissionMutation(903L,future)) {
    assertNotNull(admission); assertTrue(registry.commitItemForPublication(item,()->true));
    registry.failPrepared(item,failure);
   }
   assertFalse(future.get(2,TimeUnit.SECONDS).isSuccess());
   registry.failPrepared(item,failure);
   var events=sink.list.stream().filter(e->e.getFormattedMessage().contains("dispatch preparation failed")).toList();
   assertEquals(1,events.size(),"accepted failure must retain one diagnostic; stale repeat must not log");
   var event=events.get(0); assertEquals(Level.ERROR,event.getLevel());
   assertTrue(event.getFormattedMessage().contains("request_id=903"));
   assertTrue(event.getFormattedMessage().contains("batch_id=0"));
   assertSame(failure,((ThrowableProxy)event.getThrowableProxy()).getThrowable());
   assertNotNull(event.getThrowableProxy().getCause());
  } finally {
   if(registry.closeAdmissionAndAwaitMutations()) registry.closeOutstandingAndTerminalize();
   registry.closeExpiration();registry.closePublisher();logger.detachAppender(sink);sink.stop();logger.setLevel(previous);
  }
 }
}
