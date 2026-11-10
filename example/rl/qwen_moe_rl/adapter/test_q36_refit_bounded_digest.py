"""CPU lifecycle and byte-for-byte checks for the private digest candidate."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
import time
import unittest

from q36_refit_bounded_digest_candidate import _pipeline


class Event:
    def __init__(self, future):self.future=future
    def synchronize(self):self.future.result()


class CheckedDigest:
    def __init__(self, prefix=b'', fail_at=None):
        self.digest=hashlib.sha256(prefix);self.calls=0;self.fail_at=fail_at
        self.active=0;self.lock=threading.Lock()

    def update(self, payload):
        with self.lock:self.active+=1;self.calls+=1;call=self.calls
        try:
            expected=bytes(payload);time.sleep(.001)
            if expected!=bytes(payload):raise AssertionError('slot reused during digest')
            if call==self.fail_at:raise RuntimeError('injected hash failure')
            self.digest.update(payload)
        finally:
            with self.lock:self.active-=1


class BoundedDigestTest(unittest.TestCase):
    def run_case(self, size, chunk, fail_hash=None, fail_copy=None):
        src=bytes((i*73+19)%256 for i in range(size));slots=[bytearray(chunk),bytearray(chunk)]
        source=CheckedDigest(b'prefix\0',fail_at=fail_hash);item=CheckedDigest()
        with ThreadPoolExecutor(max_workers=1) as copies:
            def submit(slot,offset,length):
                def copy():
                    time.sleep(.001)
                    if offset==fail_copy:raise RuntimeError('injected copy failure')
                    slots[slot][:length]=src[offset:offset+length]
                return Event(copies.submit(copy))
            def view(slot,length):return memoryview(slots[slot])[:length]
            if fail_hash is not None or fail_copy is not None:
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    _pipeline(size,chunk,submit,view,source,item,serial_cutoff=0)
            else:
                _pipeline(size,chunk,submit,view,source,item,serial_cutoff=0)
                self.assertEqual(item.digest.digest(),hashlib.sha256(src).digest())
                self.assertEqual(source.digest.digest(),hashlib.sha256(b'prefix\0'+src).digest())
        self.assertEqual(source.active,0);self.assertEqual(item.active,0)

    def test_exact_digest_and_slot_lifetime(self):
        for size in (0,1,7,8,9,17,257,65539):
            with self.subTest(size=size):self.run_case(size,8 if size<300 else 4096)

    def test_failure_drains_hash_consumers(self):
        self.run_case(4096,512,fail_hash=2)
        self.run_case(4096,512,fail_copy=1024)


if __name__=='__main__':unittest.main()
