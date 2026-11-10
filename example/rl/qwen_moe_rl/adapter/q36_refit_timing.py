"""Opt-in host wall-clock observations for the existing full-HF RTP refit.

RPC duration includes network/control waits and native GPU work; it must never
be reported as pure device-copy time. No tensor payload is retained here.
"""
from contextlib import contextmanager
import time

class RefitTimings:
    def __init__(self,enabled):
        self.enabled=bool(enabled);self.start=time.monotonic();self.stages={};self.bytes=0;self.tensors=0
        self.first_next_seconds=None;self.final_next_seconds=None

    @contextmanager
    def stage(self,name):
        if not self.enabled:
            yield;return
        start=time.monotonic()
        try:yield
        finally:
            elapsed=time.monotonic()-start
            row=self.stages.setdefault(name,{'seconds':0.0,'calls':0,'max_seconds':0.0})
            row['seconds']+=elapsed;row['calls']+=1;row['max_seconds']=max(row['max_seconds'],elapsed)

    def iterate(self,values):
        it=iter(values)
        while True:
            start=time.monotonic() if self.enabled else None
            try:
                with self.stage('export_iterator_next'):
                    item=next(it)
            except StopIteration:
                if self.enabled:self.final_next_seconds=time.monotonic()-start
                return
            if self.enabled and self.tensors==0:self.first_next_seconds=time.monotonic()-start
            self.tensors+=1
            yield item

    def record_bytes(self,n):
        if self.enabled:self.bytes+=int(n)

    def snapshot(self):
        return {'scope':'host wall clock inside RTP update_weights; excludes eager exporter setup before entry; RPC includes native work/waits; stages need not sum to total',
                'elapsed_seconds':time.monotonic()-self.start,'source_tensors_seen':self.tensors,
                'export_first_next_seconds':self.first_next_seconds,'export_final_next_seconds':self.final_next_seconds,
                'acked_source_bytes':self.bytes,'stages':self.stages}
