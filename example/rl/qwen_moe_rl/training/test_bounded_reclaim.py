import hashlib,json,os,struct,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from bounded_reclaim import build_plan,advise_one_batch,ReclaimCursor

class BoundReclaimTest(unittest.TestCase):
    def fixture(self,root):
        model=root/'model';model.mkdir();run=root/'run';(run/'training').mkdir(parents=True)
        name='model-00001-of-00213.safetensors';key='ordinary'
        header=json.dumps({key:dict(dtype='BF16',shape=[32768,2],data_offsets=[0,131072])}).encode()
        (model/name).write_bytes(struct.pack('<Q',len(header))+header+bytes(range(256))*512)
        index=json.dumps({'weight_map':{key:name}}).encode();(model/'model.safetensors.index.json').write_bytes(index)
        peers=[]
        for rank in range(4):
            r=dict(rank=rank,world_size=72,hostname='host',pid=100+rank,runtime_parameters=1,source_keys=1,
             source_provider='platform_mounted',all_master_dtype='bfloat16',official_source_dtype='bfloat16',
             source_provider_close_receipt={'source':'existing_snapshot'},main_policy_source_files_consumed=1,
             started_unix=1,finished_unix=2,loaded_local_numel=16384,
             source_slices=[dict(name=key,canonical_name=key,checkpoint_file=name,source_begin=rank*8192,
              source_end=(rank+1)*8192,local_bytes=32768,source_reads=1,
              source_coordinates_verified=True,full_destination_BF16_bytes_verified=True,
              full_slice_bytes_sha256=['a'*64])])
            (run/'training'/('checkpoint-load-rank%d.json'%rank)).write_text(json.dumps(r))
            peers.append(dict(rank=rank,pid=100+rank,hostname='host'))
        return model,run,peers,hashlib.sha256(index).hexdigest(),name
    def test_plan_and_bounded_readonly_advice(self):
        with tempfile.TemporaryDirectory() as td:
            model,run,peers,index,name=self.fixture(Path(td));before=(model/name).read_bytes()
            plan=build_plan(run,peers,'host',model=model,expected_index_sha=index,expected_parameters=1,expected_sources=1)
            self.assertEqual(len(plan['ranks']),4);self.assertLessEqual(plan['union_bytes'],131072)
            with patch('os.posix_fadvise',create=True) as advise,patch('os.POSIX_FADV_DONTNEED',4,create=True):
                result=advise_one_batch(plan,plan['intervals'],model=model,max_bytes=2*plan['page_size'])
            self.assertLessEqual(result['syscall_advised_bytes'],2*plan['page_size']);self.assertGreater(advise.call_count,0)
            self.assertEqual((model/name).read_bytes(),before)
            self.assertTrue(all(x['sample_sha256_before']==x['sample_sha256_after'] for x in result['ranges']))
            with self.assertRaises(ValueError):advise_one_batch(plan,[(name,0,plan['page_size'])],model=model,max_bytes=2*plan['page_size'])
    def test_wrong_rank_pid_and_incomplete_destination_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            model,run,peers,index,_=self.fixture(Path(td));peers[0]['pid']+=1
            with self.assertRaises(ValueError):build_plan(run,peers,'host',model=model,expected_index_sha=index,expected_parameters=1,expected_sources=1)
            peers[0]['pid']-=1;p=run/'training/checkpoint-load-rank0.json';d=json.loads(p.read_text())
            d['source_slices'][0]['full_destination_BF16_bytes_verified']=False;p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):build_plan(run,peers,'host',model=model,expected_index_sha=index,expected_parameters=1,expected_sources=1)
    def test_exclusive_monotonic_cursor_and_unreplayable_claim(self):
        with tempfile.TemporaryDirectory() as td:
            model,run,peers,index,_=self.fixture(Path(td))
            plan=build_plan(run,peers,'host',model=model,expected_index_sha=index,expected_parameters=1,expected_sources=1)
            page=plan['page_size'];cursor=ReclaimCursor(plan,max_total=4*page)
            with self.assertRaises(FileExistsError):ReclaimCursor(plan,max_total=4*page)
            with patch('os.posix_fadvise',create=True),patch('os.POSIX_FADV_DONTNEED',4,create=True):
                a=cursor.next_batch(model=model,max_bytes=2*page);b=cursor.next_batch(model=model,max_bytes=2*page)
                self.assertIsNone(cursor.next_batch(model=model,max_bytes=2*page))
            self.assertEqual(a['advised_bytes'],page);self.assertEqual(b['advised_bytes'],page)
            self.assertGreaterEqual(b['ranges'][0]['offset'],a['ranges'][-1]['offset']+a['ranges'][-1]['length'])
            self.assertEqual(b['total_advised_bytes'],4*page)
    def test_stage2_sparse_window_not_rescanned_after_truncation(self):
        with tempfile.TemporaryDirectory() as td:
            model,run,peers,index,name=self.fixture(Path(td))
            plan=build_plan(run,peers,'host',model=model,expected_index_sha=index,expected_parameters=1,expected_sources=1)
            page=plan['page_size'];plan['intervals']=[(name,page,2*page)]
            plan['wider_only_intervals']=[(name,3*page,7*page)]
            cursor=ReclaimCursor(plan,max_total=8*page)
            sparse=[(name,3*page,4*page),(name,5*page,6*page)]
            with patch('os.posix_fadvise',create=True),patch('os.POSIX_FADV_DONTNEED',4,create=True),patch('bounded_reclaim.resident_runs',return_value=sparse) as scan:
                first=cursor.next_batch(model=model,max_bytes=2*page)
                second=cursor.next_batch(model=model,max_bytes=2*page)
                third=cursor.next_batch(model=model,max_bytes=2*page)
                self.assertIsNone(cursor.next_batch(model=model,max_bytes=2*page))
            self.assertEqual(first['stage'],'exact_completed_source_ranges')
            self.assertEqual(second['stage'],'wider_closed_policy_tensor_pages')
            self.assertEqual(third['stage'],'wider_closed_policy_tensor_pages')
            self.assertEqual(second['ranges'][0]['offset'],3*page)
            self.assertEqual(third['ranges'][0]['offset'],5*page)
            self.assertEqual(scan.call_count,1);self.assertEqual(cursor.wide_scanned,4*page)
            self.assertEqual(cursor.total,6*page)

if __name__=='__main__':unittest.main()
