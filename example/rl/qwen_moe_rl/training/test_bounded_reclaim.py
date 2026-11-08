import hashlib,json,os,random,struct,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from bounded_reclaim import build_plan,advise_one_batch,ReclaimCursor,MAX_WIDE_WINDOWS_PER_BATCH,MAX_ADVICE_RANGES_PER_BATCH

class BoundReclaimTest(unittest.TestCase):
    def test_exact_batch_32_range_limit_preserves_pending_and_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            model,run,peers,index,name=self.fixture(Path(td))
            plan=build_plan(run,peers,'host',model=model,expected_index_sha=index,expected_parameters=1,expected_sources=1)
            page=512;plan['page_size']=page
            plan['intervals']=[(name,(i+1)*page,(i+2)*page) for i in range(65)]
            plan['wider_only_intervals']=[]
            cursor=ReclaimCursor(plan,max_total=200*page)
            with patch('os.posix_fadvise',create=True) as advise,patch('os.POSIX_FADV_DONTNEED',4,create=True):
                batches=[cursor.next_batch(model=model,max_bytes=100*page) for _ in range(3)]
                self.assertIsNone(cursor.next_batch(model=model,max_bytes=100*page))
            self.assertEqual([b['advice_range_count'] for b in batches],[32,32,1])
            self.assertTrue(all(b['advice_range_limit']==32 for b in batches))
            self.assertEqual([r['offset'] for b in batches for r in b['ranges']],[(i+1)*page for i in range(65)])
            self.assertEqual(cursor.total,130*page)
            self.assertEqual(advise.call_count,130)

    def test_wide_batch_32_window_limit_preserves_all_pending(self):
        with tempfile.TemporaryDirectory() as td:
            page=4096;plan=self._many_intervals(Path(td),65);cursor=ReclaimCursor(plan,max_total=200*page)
            def resident(plan,name,start,end,*,model):return [(name,start,end)]
            with patch('bounded_reclaim.resident_runs',side_effect=resident) as scan,patch('bounded_reclaim.advise_one_batch',side_effect=self._fake_advice):
                batches=[cursor.next_batch(max_bytes=100*page) for _ in range(3)]
                self.assertIsNone(cursor.next_batch(max_bytes=100*page))
            self.assertEqual([b['wide_scanned_windows'] for b in batches],[32,32,1])
            self.assertTrue(all(b['wide_windows_limit']==32 for b in batches))
            self.assertEqual([r['offset'] for b in batches for r in b['ranges']],[2*i*page for i in range(65)])
            self.assertEqual(cursor.total,130*page);self.assertEqual(scan.call_count,65)

    def test_invalid_batch_limits_fail_before_scan_and_poison(self):
        for amount in (0,4096,8193,4*1024**3+4096,True):
            with self.subTest(amount=amount),tempfile.TemporaryDirectory() as td:
                cursor=ReclaimCursor(self._many_intervals(Path(td),1))
                with patch('bounded_reclaim.resident_runs') as scan:
                    with self.assertRaises(ValueError):cursor.next_batch(max_bytes=amount)
                    scan.assert_not_called()
                    with self.assertRaisesRegex(RuntimeError,'forbids all replay'):cursor.next_batch()

    def test_wrong_process_and_failure_marker_block_owner_before_advice(self):
        with tempfile.TemporaryDirectory() as td:
            plan=self._many_intervals(Path(td),1);cursor=ReclaimCursor(plan)
            with patch('bounded_reclaim.resident_runs') as scan,patch('bounded_reclaim.os.getpid',return_value=cursor._owner_pid+1):
                with self.assertRaisesRegex(RuntimeError,'original coordinator'):cursor.next_batch()
                scan.assert_not_called()
            with patch('bounded_reclaim.resident_runs') as scan:
                with self.assertRaisesRegex(RuntimeError,'forbids all replay'):cursor.next_batch()
                scan.assert_not_called()
            self.assertTrue((cursor.directory/'FAILED.json').is_file())

    def test_replaced_or_modified_claim_fails_before_advice(self):
        for mode in ('replace','mutate','remove'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                plan=self._many_intervals(Path(td),1);cursor=ReclaimCursor(plan);claim=cursor.directory/'CLAIM.json'
                if mode=='replace':
                    raw=claim.read_bytes();replacement=cursor.directory/'replacement';replacement.write_bytes(raw);os.replace(replacement,claim)
                elif mode=='mutate':claim.write_text('changed')
                else:claim.unlink()
                with patch('bounded_reclaim.resident_runs') as scan:
                    with self.assertRaises((OSError,RuntimeError)):cursor.next_batch()
                    scan.assert_not_called()
                    with self.assertRaisesRegex(RuntimeError,'forbids all replay'):cursor.next_batch()
                self.assertTrue((cursor.directory/'FAILED.json').is_file())

    def test_preexisting_failure_marker_blocks_original_owner(self):
        with tempfile.TemporaryDirectory() as td:
            cursor=ReclaimCursor(self._many_intervals(Path(td),1))
            (cursor.directory/'FAILED.json').write_text('{}')
            with patch('bounded_reclaim.resident_runs') as scan:
                with self.assertRaisesRegex(RuntimeError,'failure marker'):cursor.next_batch()
                scan.assert_not_called()

    def test_randomized_wide_coverage_and_accounting_64k(self):
        randomizer=random.Random(20261108);page=65536
        for case in range(120):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as td:
                plan=self._many_intervals(Path(td),0);plan['page_size']=page
                position=0;expected=[]
                for _ in range(randomizer.randint(1,80)):
                    position+=randomizer.randint(0,3)*page
                    size=randomizer.randint(1,8)*page
                    plan['wider_only_intervals'].append(('f',position,position+size))
                    for off in range(position,position+size,page):
                        if randomizer.randrange(2):expected.append(off)
                    position+=size
                resident_set=set(expected)
                def resident(plan,name,start,end,*,model):
                    return [(name,off,off+page) for off in range(start,end,page) if off in resident_set]
                cursor=ReclaimCursor(plan,max_total=2048*page);seen=[]
                with patch('bounded_reclaim.resident_runs',side_effect=resident),patch('bounded_reclaim.advise_one_batch',side_effect=self._fake_advice):
                    while True:
                        batch=cursor.next_batch(max_bytes=17*page)
                        if batch is None:break
                        self.assertLessEqual(batch['wide_scanned_windows'],32)
                        self.assertLessEqual(batch['wide_scanned_bytes'],2*1024**3)
                        self.assertLessEqual(batch['syscall_advised_bytes'],17*page)
                        for row in batch['ranges']:seen.extend(range(row['offset'],row['offset']+row['length'],page))
                self.assertEqual(seen,expected)
                self.assertEqual(cursor.total,2*len(expected)*page)
                self.assertLessEqual(cursor.total,2048*page)

    def _many_intervals(self, root, count):
        page=4096;run=root/'run';run.mkdir()
        return dict(run_root=str(run),run_id='run',hostname='host',ranks=[],page_size=page,
                    intervals=[],wider_only_intervals=[('f',2*i*page,(2*i+1)*page) for i in range(count)])

    @staticmethod
    def _fake_advice(plan, ranges, *, model, max_bytes):
        page=plan['page_size'];items=[];remaining=max_bytes
        for name,start,end in ranges:
            if remaining<2*page or len(items)>=MAX_ADVICE_RANGES_PER_BATCH:break
            size=min(end-start,remaining-page);size-=size%page
            if size:items.append(dict(file=name,offset=start,length=size));remaining-=size+page
        n=sum(row['length'] for row in items)
        return dict(advised_bytes=n,ranges=items,syscall_advised_bytes=n+len(items)*page)

    def test_many_tiny_intervals_are_scanned_in_bounded_rounds(self):
        with tempfile.TemporaryDirectory() as td:
            plan=self._many_intervals(Path(td),1100);cursor=ReclaimCursor(plan,max_total=4*1024**3)
            with patch('bounded_reclaim.resident_runs',return_value=[]) as scan,patch('bounded_reclaim.advise_one_batch',side_effect=self._fake_advice):
                batches=[cursor.next_batch() for _ in range(35)]
                self.assertIsNone(cursor.next_batch())
            self.assertEqual([b['wide_scanned_windows'] for b in batches],[32]*34+[12])
            self.assertEqual(sum(b['wide_scanned_bytes'] for b in batches),1100*4096)
            self.assertEqual(scan.call_count,1100);self.assertEqual(cursor.total,0)
            self.assertTrue(all(b['syscall_advised_bytes']<=4*1024**3 for b in batches))

    def test_many_resident_intervals_keep_partial_queue_after_final_scan(self):
        with tempfile.TemporaryDirectory() as td:
            plan=self._many_intervals(Path(td),5);cursor=ReclaimCursor(plan,max_total=20*4096)
            def resident(plan,name,start,end,*,model):return [(name,start,end)]
            with patch('bounded_reclaim.resident_runs',side_effect=resident) as scan,patch('bounded_reclaim.advise_one_batch',side_effect=self._fake_advice):
                batches=[cursor.next_batch(max_bytes=4*4096) for _ in range(3)]
                self.assertIsNone(cursor.next_batch(max_bytes=4*4096))
            self.assertEqual([len(b['ranges']) for b in batches],[2,2,1])
            self.assertEqual([b['wide_scanned_windows'] for b in batches],[5,0,0])
            self.assertEqual(scan.call_count,5)
            self.assertEqual([r['offset'] for b in batches for r in b['ranges']],[i*8192 for i in range(5)])
            self.assertEqual(cursor.total,10*4096)

    def test_scan_failure_forbids_new_cursor_claim(self):
        with tempfile.TemporaryDirectory() as td:
            plan=self._many_intervals(Path(td),3);cursor=ReclaimCursor(plan)
            with patch('bounded_reclaim.resident_runs',side_effect=[[],OSError('failed')]):
                with self.assertRaises(OSError):cursor.next_batch()
                with self.assertRaisesRegex(RuntimeError,'forbids all replay'):cursor.next_batch()
            self.assertTrue((cursor.directory/'FAILED.json').is_file())
            with self.assertRaises(FileExistsError):ReclaimCursor(plan)

    def test_partial_advice_failure_poisoned_same_cursor_and_claim(self):
        with tempfile.TemporaryDirectory() as td:
            model,run,peers,index,name=self.fixture(Path(td))
            plan=build_plan(run,peers,'host',model=model,expected_index_sha=index,expected_parameters=1,expected_sources=1)
            page=plan['page_size'];plan['intervals']=[(name,page,2*page),(name,3*page,4*page)];plan['wider_only_intervals']=[]
            cursor=ReclaimCursor(plan,max_total=8*page)
            with patch('os.POSIX_FADV_DONTNEED',4,create=True),patch('os.posix_fadvise',side_effect=[None,None,OSError('second range failed')],create=True) as advise:
                with self.assertRaises(OSError):cursor.next_batch(model=model,max_bytes=8*page)
                self.assertEqual(advise.call_count,3)
                with self.assertRaisesRegex(RuntimeError,'forbids all replay'):cursor.next_batch(model=model,max_bytes=8*page)
                self.assertEqual(advise.call_count,3)
            failure=json.loads((cursor.directory/'FAILED.json').read_text())
            self.assertTrue(failure['partial_current_batch_may_have_executed'])
            self.assertEqual(failure['completed_syscall_advised_bytes'],0)
            with self.assertRaises(FileExistsError):ReclaimCursor(plan,max_total=8*page)

    def test_receipt_write_failure_after_advice_poisoned_even_if_failure_record_fails(self):
        with tempfile.TemporaryDirectory() as td:
            model,run,peers,index,name=self.fixture(Path(td))
            plan=build_plan(run,peers,'host',model=model,expected_index_sha=index,expected_parameters=1,expected_sources=1)
            page=plan['page_size'];plan['intervals']=[(name,page,2*page)];plan['wider_only_intervals']=[]
            cursor=ReclaimCursor(plan,max_total=4*page)
            with patch('os.POSIX_FADV_DONTNEED',4,create=True),patch('os.posix_fadvise',create=True) as advise,patch('bounded_reclaim.json.dump',side_effect=OSError('receipt write failed')):
                with self.assertRaises(OSError):cursor.next_batch(model=model,max_bytes=2*page)
                self.assertEqual(advise.call_count,2)
                with self.assertRaisesRegex(RuntimeError,'forbids all replay'):cursor.next_batch(model=model,max_bytes=2*page)
                self.assertEqual(advise.call_count,2)
            with self.assertRaises(FileExistsError):ReclaimCursor(plan,max_total=4*page)

    def test_large_single_interval_scans_at_most_two_gib_per_round(self):
        with tempfile.TemporaryDirectory() as td:
            plan=self._many_intervals(Path(td),0);plan['wider_only_intervals']=[('f',0,3*1024**3)]
            cursor=ReclaimCursor(plan)
            with patch('bounded_reclaim.resident_runs',return_value=[]) as scan,patch('bounded_reclaim.advise_one_batch',side_effect=self._fake_advice):
                a=cursor.next_batch();b=cursor.next_batch();self.assertIsNone(cursor.next_batch())
            self.assertEqual([a['wide_scanned_bytes'],b['wide_scanned_bytes']],[2*1024**3,1024**3])
            self.assertEqual(scan.call_count,2);self.assertEqual(cursor.wide_scanned,3*1024**3)

    def test_selector_outside_current_window_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            plan=self._many_intervals(Path(td),2);cursor=ReclaimCursor(plan)
            with patch('bounded_reclaim.resident_runs',return_value=[('f',8192,12288)]):
                with self.assertRaisesRegex(RuntimeError,'escaped'):cursor.next_batch()
            self.assertEqual(cursor.wide_scanned,0)
            with self.assertRaises(FileExistsError):ReclaimCursor(plan)

    def test_64k_pages_use_actual_advice_accounting(self):
        with tempfile.TemporaryDirectory() as td:
            model,run,peers,index,name=self.fixture(Path(td))
            plan=build_plan(run,peers,'host',model=model,expected_index_sha=index,expected_parameters=1,expected_sources=1)
            page=65536;plan['page_size']=page;plan['intervals']=[(name,page,2*page)];plan['wider_only_intervals']=[]
            with patch('os.posix_fadvise',create=True) as advise,patch('os.POSIX_FADV_DONTNEED',4,create=True):
                result=advise_one_batch(plan,plan['intervals'],model=model,max_bytes=2*page)
            self.assertEqual(advise.call_count,2);self.assertEqual(result['sample_read_bytes'],2*page)
            self.assertEqual(result['syscall_advised_bytes'],2*page)
            self.assertEqual(result['ranges'][0]['sample_sha256_before'],result['ranges'][0]['sample_sha256_after'])

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
