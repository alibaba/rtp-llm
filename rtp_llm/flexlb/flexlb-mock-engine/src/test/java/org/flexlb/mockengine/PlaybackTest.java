package org.flexlb.mockengine;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import org.junit.jupiter.api.Test;

class PlaybackTest {
    @Test void clearedLauncherEnvironmentUsesDefaults() {
        Playback p=new Playback(Map.of("MAX_LAPS","","LAP_IDENTITY","","PLAYBACK_SEED","","BURST_FACTOR",""),false,30);
        assertEquals(1,p.maxLaps);assertEquals("structural-relabel",p.identity);assertEquals(1,p.burstFactor);
    }
    @Test void uniformDoesNotImplyLoopAndLoopRequiresBound() {
        Playback once=new Playback(Map.of(),false,30);
        assertFalse(once.more(0));
        assertThrows(IllegalArgumentException.class,()->new Playback(Map.of(),true,0));
        Playback twice=new Playback(Map.of("MAX_LAPS","2"),false,0);
        assertTrue(twice.more(0));assertFalse(twice.more(1));
    }
    @Test void structuralRelabelPreservesWithinLapPrefixesAndChangesActualKeys() {
        Playback p=new Playback(Map.of(),true,30);
        List<Integer> a=new ArrayList<>(Collections.nCopies(512,1));a.addAll(Collections.nCopies(512,2));
        List<Integer> b=new ArrayList<>(Collections.nCopies(512,1));b.addAll(Collections.nCopies(512,3));
        List<Long> ak=JavaLoadClient.computeBlockKeys(a,512),bk=JavaLoadClient.computeBlockKeys(b,512);
        List<Integer> a1=p.relabel(a,ak,512,1,4), b1=p.relabel(b,bk,512,1,4);
        List<Long> a1k=JavaLoadClient.computeBlockKeys(a1,512),b1k=JavaLoadClient.computeBlockKeys(b1,512);
        assertEquals(a1k.get(0),b1k.get(0));assertNotEquals(a1k.get(1),b1k.get(1));
        assertTrue(Collections.disjoint(ak,a1k));assertEquals(1,a.get(0));
        assertEquals(a1,p.relabel(a,ak,512,1,4));
        assertThrows(IllegalArgumentException.class,()->p.relabel(a,ak,512,2,Integer.MAX_VALUE));
    }
    @Test void partialEndpointsHaveExactSemantics() {
        List<Integer> tokens=Collections.nCopies(512,7);List<Long> keys=JavaLoadClient.computeBlockKeys(tokens,512);
        Playback none=new Playback(Map.of("LAP_IDENTITY","none"),true,30);
        assertSame(tokens,none.relabel(tokens,keys,512,3,8));
        Playback keep=new Playback(Map.of("LAP_IDENTITY","partial","LAP_RETAIN_PROBABILITY","1"),true,30);
        assertEquals(tokens,keep.relabel(tokens,keys,512,3,8));
        Playback fresh=new Playback(Map.of("LAP_IDENTITY","partial","LAP_RETAIN_PROBABILITY","0"),true,30);
        assertEquals(31,fresh.relabel(tokens,keys,512,3,8).get(0));
    }
    @Test void uniformRampBurstAndWaveHaveMonotoneIntegratedSchedules() {
        Playback p=new Playback(Map.of(),false,30);
        assertEquals(2,p.new Clock().due(20,10,0),1e-9);
        assertEquals(2,p.new Clock().due(10,10,2),1e-9);
        Playback burst=new Playback(Map.of("BURST_FACTOR","4","BURST_PERIOD_SECONDS","10","BURST_DUTY","0.25"),false,30);
        Playback.Clock clock=burst.new Clock();
        double first=clock.due(10,10,0), end=clock.due(100,10,0);
        assertEquals(0.4375,first,0.001);assertEquals(10,end,0.001);
        Playback wave=new Playback(Map.of("DIURNAL_AMPLITUDE","0.5","DIURNAL_PERIOD_SECONDS","20"),false,30);
        Playback.Clock wc=wave.new Clock();double prev=-1;
        for(int i=0;i<200;i++){double t=wc.due(i,10,0);assertTrue(t>prev);prev=t;}
        assertEquals(20,wc.due(200,10,0),0.001);
    }
    @Test void pinnedBlocksAreDecodedExactly() throws Exception {
        com.fasterxml.jackson.databind.ObjectMapper mapper=new com.fasterxml.jackson.databind.ObjectMapper();
        var row=mapper.createObjectNode();row.put("il",1025);row.put("cache_key_block_size",512);
        var blocks=row.putArray("input_token_blocks");var pinned=blocks.addArray();
        for(int i=0;i<512;i++)pinned.add(i);
        blocks.add(99);blocks.add(7);
        List<Integer> tokens=JavaLoadClient.decodeInputTokens(row,1025);
        assertEquals(511,tokens.get(511));assertEquals(99,tokens.get(512));assertEquals(7,tokens.get(1024));
        assertEquals(1025,tokens.size());
    }
}
