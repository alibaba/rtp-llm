import copy
import unittest
from cpu_admission import GIB,evaluate

RUN='q38-test';HOST='test-host';CID='test-container';NOW=1800000000;MONO=10000

def stat(**changes):
    d=dict(anon=300*GIB,file=20*GIB,inactive_file=10*GIB,file_dirty=GIB,file_writeback=GIB,
           file_mapped=GIB,shmem=GIB,unevictable=0)
    d.update(changes);return d

def snapshot():
    row=dict(path='/pod/container',is_exact_container=True,identity_verified=True,
             limit_bytes=800*GIB,current_bytes=300*GIB,stat=stat(),
             events_delta={'oom':0,'oom_kill':0},pressure='some avg10=0')
    parent=copy.deepcopy(row);parent.update(path='/pod',is_exact_container=False)
    return dict(run_id=RUN,hostname=HOST,container_id=CID,identity_verified=True,
                same_time_namespace=True,time_namespace='time:[1]',sample_started_monotonic=MONO-1,
                sample_completed_monotonic=MONO-.5,sample_unix=NOW-1,cgroups=[row,parent])

def peers():
    return [dict(rank=r,hostname=HOST,cpu_gradient_bytes=20*GIB,
                 allocated_moment_bytes=0,planned_moment_bytes=40*GIB) for r in range(4)]

def decide(s=None,p=None):
    return evaluate(s or snapshot(),p or peers(),run_id=RUN,hostname=HOST,
                    expected_cid=CID,expected_time_namespace='time:[1]',now=NOW,monotonic_now=MONO)

class AdmissionTest(unittest.TestCase):
    def test_current_already_includes_non_state_reserve(self):
        result=decide();self.assertTrue(result['passed'])
        for row in result['cgroups']:
            self.assertEqual(row['reclaimable_inactive_upper_bytes'],6*GIB)
            self.assertEqual(row['credited_cache_bytes'],0)
            self.assertEqual(row['non_state_charged_bytes'],220*GIB)
            self.assertEqual(row['reserve_shortfall_bytes'],0)
            self.assertEqual(row['required_bytes'],460*GIB)
    def test_only_missing_reserve_is_added(self):
        s=snapshot()
        for row in s['cgroups']:
            row['current_bytes']=120*GIB;row['stat']=stat(anon=120*GIB,file=0,inactive_file=0,file_dirty=0,
                file_writeback=0,file_mapped=0,shmem=0,unevictable=0)
        row=decide(s)['cgroups'][0]
        self.assertEqual(row['non_state_charged_bytes'],40*GIB)
        self.assertEqual(row['reserve_shortfall_bytes'],20*GIB)
        self.assertEqual(row['required_bytes'],300*GIB)
    def test_parent_tighter_than_container_denies_everyone(self):
        s=snapshot();s['cgroups'][1]['limit_bytes']=400*GIB
        self.assertFalse(decide(s)['passed']);self.assertFalse(decide(s)['cgroups'][1]['passed'])
    def test_active_file_is_not_credit(self):
        s=snapshot()
        for row in s['cgroups']:
            row['stat']=stat(file=500*GIB,inactive_file=0,file_dirty=0,file_writeback=0,
                             file_mapped=0,shmem=0,unevictable=0)
        self.assertEqual(decide(s)['cgroups'][0]['reclaimable_inactive_upper_bytes'],0)
    def test_file_cache_never_offsets_forward_reserve(self):
        s=snapshot()
        for row in s['cgroups']:
            row['current_bytes']=300*GIB;row['stat']=stat(anon=80*GIB,file=220*GIB,
                inactive_file=100*GIB,file_dirty=0,file_writeback=0,file_mapped=0,shmem=0,unevictable=0)
        row=decide(s)['cgroups'][0]
        self.assertEqual(row['non_state_charged_bytes'],0)
        self.assertEqual(row['reserve_shortfall_bytes'],60*GIB)
        self.assertEqual(row['required_bytes'],520*GIB)
    def test_missing_stale_wrong_identity_or_partial_moments_rejected(self):
        for change in ('stale','future','cid','hostname','identity','peers','moments','stat','oom','pressure','time_ns','mono_stale'):
            with self.subTest(change=change):
                s=snapshot();p=peers()
                if change=='stale':s['sample_unix']=NOW-16
                if change=='future':s['sample_unix']=NOW+1
                if change=='cid':s['container_id']='wrong'
                if change=='hostname':s['hostname']='wrong'
                if change=='identity':s['cgroups'][1]['identity_verified']=False
                if change=='peers':p[3]['rank']=0
                if change=='moments':p[0]['allocated_moment_bytes']=1
                if change=='stat':del s['cgroups'][0]['stat']['file_mapped']
                if change=='oom':s['cgroups'][1]['events_delta']['oom']=1
                if change=='pressure':s['cgroups'][0]['pressure']=None
                if change=='time_ns':s['time_namespace']='time:[2]'
                if change=='mono_stale':s['sample_started_monotonic']=MONO-16
                with self.assertRaises((ValueError,KeyError)):decide(s,p)
    def test_large_pending_moments_deny_when_file_cache_dominates(self):
        s=snapshot()
        for group in s['cgroups']:
            group['current_bytes']=597*GIB
            group['stat']=stat(anon=236*GIB,file=361*GIB,inactive_file=200*GIB,
                file_dirty=0,file_writeback=0,file_mapped=0,shmem=0,unevictable=0)
        p=[dict(rank=rank,hostname=HOST,cpu_gradient_bytes=49*GIB,
                allocated_moment_bytes=0,planned_moment_bytes=124*GIB) for rank in range(4)]
        result=decide(s,p)
        self.assertFalse(result['passed'])
        self.assertGreater(result['cgroups'][0]['required_bytes'],800*GIB)
        self.assertFalse(result['active_file_credited'])

if __name__=='__main__':unittest.main()
