import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from bounded_reclaim import PinnedSources,ReclaimCursor,build_plan,_open_verified
from test_bounded_reclaim import BoundReclaimTest

class PinnedReclaimTest(unittest.TestCase):
    def setup_plan(self,root):
        model,run,peers,index,name=BoundReclaimTest().fixture(root)
        # The production process is Linux and requires /proc/self/fd; tests
        # run on the development Mac as well as Linux.
        with patch('bounded_reclaim.os.listdir',return_value=[]):pin=PinnedSources(model)
        plan=build_plan(run,peers,'host',model=model,expected_index_sha=index,
                        expected_parameters=1,expected_sources=1,pinned=pin)
        return model,run,name,pin,plan

    def test_rename_path_keeps_original_fd_and_close(self):
        with tempfile.TemporaryDirectory() as td:
            model,_,name,pin,plan=self.setup_plan(Path(td));fd=pin.borrow(name,plan['files'])
            old=os.pread(fd,256,0);replacement=model/'replacement';replacement.write_bytes(b'x'*1024)
            os.replace(replacement,model/name)
            with patch('bounded_reclaim.os.open',side_effect=AssertionError('source must not reopen')):
                checked,_=_open_verified(plan,model,name,pinned=pin)
                self.assertEqual(checked,fd);self.assertEqual(os.pread(checked,256,0),old)
            pin.close()
            with self.assertRaises(OSError):os.fstat(fd)

    def test_changed_header_and_plan_metadata_fail_closed(self):
        for mode in ('header','metadata'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                model,_,name,pin,plan=self.setup_plan(Path(td));fd=pin.borrow(name,plan['files'])
                if mode=='header':
                    with (model/name).open('r+b') as f:f.seek(10);f.write(b'X')
                else:plan['files'][name]['header_sha256']='f'*64
                with self.assertRaisesRegex(ValueError,'header changed|metadata changed'):_open_verified(plan,model,name,pinned=pin)
                pin.close()
                with self.assertRaises(OSError):os.fstat(fd)

    def test_two_batches_share_same_pinned_fd_and_total(self):
        with tempfile.TemporaryDirectory() as td:
            model,_,name,pin,plan=self.setup_plan(Path(td));fd=pin.borrow(name,plan['files'])
            page=plan['page_size'];cursor=ReclaimCursor(plan,pinned=pin,max_total=4*page)
            with patch('os.POSIX_FADV_DONTNEED',4,create=True),patch('os.posix_fadvise',create=True):
                a=cursor.next_batch(model=model,max_bytes=2*page)
                self.assertEqual(pin.borrow(name,plan['files']),fd)
                b=cursor.next_batch(model=model,max_bytes=2*page)
            self.assertEqual((a['round'],b['round'],cursor.total),(1,2,4*page))
            cursor.close()
            with self.assertRaises(OSError):os.fstat(fd)
            with self.assertRaisesRegex(RuntimeError,'forbids all replay'):cursor.next_batch(model=model)

    def test_failure_preserves_whitelisted_diagnostic_redacts_unknown_and_closes(self):
        for error,message in [(ValueError('Checkpoint header changed'),'Checkpoint header changed'),
                              (ValueError('token=SECRET payload'),None)]:
            with self.subTest(message=message),tempfile.TemporaryDirectory() as td:
                model,_,name,pin,plan=self.setup_plan(Path(td));fd=pin.borrow(name,plan['files'])
                cursor=ReclaimCursor(plan,pinned=pin)
                with patch('bounded_reclaim.advise_one_batch',side_effect=error):
                    with self.assertRaises(ValueError) as raised:cursor.next_batch(model=model)
                    self.assertIs(raised.exception,error)
                    with self.assertRaisesRegex(RuntimeError,'forbids all replay'):cursor.next_batch(model=model)
                raw=(cursor.directory/'FAILED.json').read_text();failure=json.loads(raw)
                self.assertEqual(failure['failure_message'],message)
                self.assertEqual(failure['failure_message_redacted'],message is None)
                self.assertNotIn('SECRET',raw);self.assertNotIn('payload',raw)
                self.assertTrue(failure['failure_frames']);self.assertTrue(failure['partial_current_batch_may_have_executed'])
                with self.assertRaises(OSError):os.fstat(fd)

    def test_owner_and_fd_budget_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            model,_,name,pin,plan=self.setup_plan(Path(td));fd=pin.borrow(name,plan['files'])
            with patch('bounded_reclaim.os.getpid',return_value=os.getpid()+1):
                with self.assertRaisesRegex(RuntimeError,'coordinator identity'):pin.borrow(name,plan['files'])
            pin.close()
            with self.assertRaises(OSError):os.fstat(fd)
        with patch('bounded_reclaim.os.listdir',return_value=[]),patch('bounded_reclaim.resource.getrlimit',return_value=(100,100)):
            with self.assertRaisesRegex(RuntimeError,'Insufficient file descriptor reserve'):PinnedSources()

if __name__=='__main__':unittest.main()
