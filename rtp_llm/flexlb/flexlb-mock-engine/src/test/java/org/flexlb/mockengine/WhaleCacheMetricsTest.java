package org.flexlb.mockengine;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.flexlb.engine.grpc.EngineRpcService;
import java.nio.file.*;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;
class WhaleCacheMetricsTest {
 @TempDir Path dir;
 @Test void nativeHashesMatchCppAndReuseOnlyCompleteBlocks() throws Exception {
  Path file=dir.resolve("perf.json");Files.writeString(file,"{\"block_size\":2}");
  Path master=dir.resolve("master.json");MockMasterConfig.writeWithPrefillExpression(master,"20");
  var model=MockPerformanceModel.load(file.toString(),master.toString());
  var input=EngineRpcService.GenerateInputPB.newBuilder().addAllTokenIds(List.of(1,2,3,4,5)).build();
  var cache=new MockLruBlockCache(10);
  assertTrue(model.shape(input,cache).blockKeys().isEmpty());
  model.nativeTokenCacheKeys=true;
  var cold=model.shape(input,cache);
  // Compiled against production HashUtil.h, including signed right shift.
  assertEquals(List.of(-8366447769517741626L,455111481605203084L),cold.blockKeys());
  try(var cluster=MockEngineTestCluster.create(model,61400,1,0)) {
   var demand=JavaMockEngineCluster.FastRpcService.class.getDeclaredMethod("needBlocks",MockPerformanceModel.RequestShape.class);
   demand.setAccessible(true);assertEquals(3,demand.invoke(cluster.prefill(0),cold));
  }
  cache.admit(cold.blockKeys());assertEquals(4,model.shape(input,cache).hitTokens());
  var explicit=input.toBuilder().setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder().setUniqueKey("{\"block_cache_keys\":[]}")).build();
  assertTrue(model.shape(explicit,cache).blockKeys().isEmpty());
  assertEquals(cold.blockKeys(),model.forEngine().shape(input,cache).blockKeys());
 }
 @Test void evictionLifetimeReportsOnlyRemovedBlocks() {
  var cache=new MockLruBlockCache(4,0);List<Double> times=new ArrayList<>();
  cache.setEvictionLifetimeListener(times::add);cache.admit(List.of(1L,2L));
  cache.evict(List.of(77L));assertTrue(times.isEmpty());cache.setRetentionBlocks(0);
  assertEquals(2,cache.evictions());assertEquals(2,times.size());assertTrue(times.stream().allMatch(v->v>=0));
 }
}
