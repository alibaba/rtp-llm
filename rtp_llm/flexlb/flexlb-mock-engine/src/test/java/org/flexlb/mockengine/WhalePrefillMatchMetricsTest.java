package org.flexlb.mockengine;
import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;
class WhalePrefillMatchMetricsTest {
 @Test void matchesHistoricalWindowAndKeepsPhysicalReuseDistinct() {
  var m=new WhalePrefillMatchMetrics(100);
  var cold=m.record(List.of(1L,2L),2500,0,1024,0);
  assertEquals(0.0,cold.get("rtp_llm_prefill_worker_recent_cache_key_hit_ratio").doubleValue());
  var hot=m.record(List.of(1L,2L),2500,0,1024,50);
  assertEquals(2048.0/2500,hot.get("rtp_llm_prefill_worker_recent_cache_key_hit_ratio").doubleValue());
  assertEquals(0.0,hot.get("mock_prefill_kv_match_ratio").doubleValue());
  var expired=m.record(List.of(1L,2L),2500,1024,1024,150);
  assertEquals(0.0,expired.get("rtp_llm_prefill_worker_recent_cache_key_hit_ratio").doubleValue());
  assertEquals(1024.0/2500,expired.get("mock_prefill_kv_match_ratio").doubleValue());
  assertEquals(2048.0/7500,expired.get("rtp_llm_prefill_worker_theory_cache_all_hit_ratio").doubleValue());
 }
 @Test void duplicatesWithinOneRequestAreNotPriorHistoryAndEmptyInputIsFinite() {
  var m=new WhalePrefillMatchMetrics(100);
  assertEquals(0L,m.record(List.of(1L,1L),2048,0,1024,0).get("rtp_llm_prefill_worker_recent_cache_key_hit_count"));
  var hit=m.record(List.of(1L,1L),2048,0,1024,1);
  assertEquals(2048L,hit.get("rtp_llm_prefill_worker_recent_cache_key_hit_count"));
  assertEquals(0.0,m.record(List.of(),0,0,1024,102).get("mock_prefill_kv_match_ratio").doubleValue());
 }
}
