"""Exact measured-work windows for PP Agent training; no engine scheduling.

The existing workload snapshot stores predicted remaining request work. It cannot
represent partial work executed across a dynamic PP window. These private
integer prefixes add that missing representation without a lifetime request map.
The caller owns request progress and Native step/checkpoint lifetimes.
"""
from __future__ import annotations
from bisect import bisect_right
import heapq
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Iterable


class NativeStartOrder:
    """Count later arrivals starting while an earlier arrival has not started.

    The caller uses the existing execution lock. Remove the complete first-start
    set before comparing: requests sharing a Native step have no start order.
    Lazy heap removal is compacted by live size, never by request age.
    """
    def __init__(self):
        self._pending = {}
        self._heap = []

    def admit(self, request_ids, received_ns):
        ids = tuple(request_ids)
        if type(received_ns) is not int or received_ns <= 0 or len(set(ids)) != len(ids) or any(r in self._pending for r in ids):
            raise ValueError('invalid or duplicate first-start admission')
        for rid in ids:
            self._pending[rid] = received_ns
            heapq.heappush(self._heap, (received_ns, rid))

    def _prune(self):
        while self._heap and self._pending.get(self._heap[0][1]) != self._heap[0][0]:
            heapq.heappop(self._heap)
        if len(self._heap) > 2 * len(self._pending) + 64:
            self._heap = [(clock, rid) for rid, clock in self._pending.items()]
            heapq.heapify(self._heap)

    def start_step(self, request_ids):
        clocks, unknown = {}, []
        for rid in dict.fromkeys(request_ids):
            received_ns = self._pending.pop(rid, None)
            if received_ns is None:
                unknown.append(rid)
            else:
                clocks[rid] = received_ns
        self._prune()
        oldest = self._heap[0] if self._heap else None
        overtakers = tuple(rid for rid, received_ns in clocks.items()
                           if oldest is not None and received_ns > oldest[0])
        return overtakers, oldest, tuple(unknown)

    def cancel(self, request_ids):
        for rid in request_ids:
            self._pending.pop(rid, None)
        self._prune()

@dataclass(frozen=True)
class NativeWorkPrefix:
    step_count: int = 0
    request_count: int = 0
    context_tokens: int = 0
    compute_tokens: int = 0
    compute_token_square_sum: int = 0
    context_compute_token_product_sum: int = 0

@dataclass
class NativeRequestProgress:
    initial_context_tokens: int | None = None
    _steps: list[int] = field(default_factory=list,repr=False)
    _tokens: list[int] = field(default_factory=list,repr=False)

    @property
    def compute_tokens(self):
        return self._tokens[-1] if self._tokens else 0

    def at(self,step):
        index=bisect_right(self._steps,step)-1
        return self._tokens[index] if index>=0 else 0

    def append(self,step,context,tokens):
        if type(step)is not int or step<=0 or type(context)is not int or context<0 or type(tokens)is not int or tokens<=0:
            raise ValueError('Native prefill position must use positive integer tokens and step')
        previous=self.compute_tokens
        if self.initial_context_tokens is None:
            self.initial_context_tokens=context
        if context!=self.initial_context_tokens+previous:
            raise ValueError('Native token span does not continue previous dispatch')
        if self._steps and step<self._steps[-1]:
            raise ValueError('Native step order moved backwards')
        if self._steps and step==self._steps[-1]:
            self._tokens[-1]=previous+tokens
        else:
            self._steps.append(step);self._tokens.append(previous+tokens)
        return previous

@dataclass(frozen=True)
class NativeWorkWindow:
    begin: NativeWorkPrefix
    # Only requests already started before begin can need a moment correction.
    old_requests: tuple[NativeRequestProgress,...]

    @classmethod
    def capture(cls,begin,request_refs:Iterable[NativeRequestProgress]):
        seen=set();old=[]
        for request in request_refs:
            key=id(request)
            if key not in seen:
                seen.add(key)
                if request.at(begin.step_count)>0:old.append(request)
        return cls(begin,tuple(old))

    def project(self,end):
        if end.step_count<self.begin.step_count:
            raise ValueError('Native terminal checkpoint precedes admission checkpoint')
        count=end.request_count-self.begin.request_count
        context=end.context_tokens-self.begin.context_tokens
        compute=end.compute_tokens-self.begin.compute_tokens
        square=end.compute_token_square_sum-self.begin.compute_token_square_sum
        product=end.context_compute_token_product_sum-self.begin.context_compute_token_product_sum
        for request in self.old_requests:
            before=request.at(self.begin.step_count)
            used=request.at(end.step_count)-before
            if used:
                # Global prefixes store whole-request cumulative moments. For an
                # old request remove the cross term from (before+used)^2, and
                # move its already-computed tokens into the window's context.
                count+=1;context+=request.initial_context_tokens+before
                square-=2*before*used;product+=before*used
        if min(count,context,compute,square,product)<0:
            raise ValueError('Native work window lacks an admission predecessor reference')
        return NativeWorkPrefix(end.step_count-self.begin.step_count,count,context,compute,square,product)

class NativeWorkTracker:
    """O(scheduled requests) per dispatch; owns no request or step history."""
    __slots__=('prefix',)
    def __init__(self):self.prefix=NativeWorkPrefix()

    def dispatch(self,spans:Sequence[tuple[NativeRequestProgress,int,int]]):
        if not isinstance(spans,Sequence):
            raise TypeError('Native spans must be a replayable sequence')
        # Validate the entire step before committing any request progress.
        # This is one linear pass, including repeated spans of the same request.
        expected={}
        for request,context,tokens in spans:
            if type(context)is not int or context<0 or type(tokens)is not int or tokens<=0:
                raise ValueError('Native span must use positive integer tokens')
            key=id(request);position=expected.get(key)
            if position is None:
                position=context if request.initial_context_tokens is None else request.initial_context_tokens+request.compute_tokens
            if context!=position:raise ValueError('Native token span is not contiguous')
            expected[key]=position+tokens
        if not expected:return self.prefix
        old=self.prefix;step=old.step_count+1;count=old.request_count;context_total=old.context_tokens
        compute=old.compute_tokens;square=old.compute_token_square_sum;product=old.context_compute_token_product_sum
        for request,context,tokens in spans:
            previous=request.append(step,context,tokens)
            if previous==0:
                count+=1;context_total+=request.initial_context_tokens
            compute+=tokens;square+=2*previous*tokens+tokens*tokens;product+=request.initial_context_tokens*tokens
        self.prefix=NativeWorkPrefix(step,count,context_total,compute,square,product)
        return self.prefix


@dataclass(frozen=True)
class CompletedWorkCheckpoint:
    completed_ns: int
    prefix: NativeWorkPrefix
    request_refs: tuple[NativeRequestProgress, ...] = ()


class NativeWorkHistory:
    """Keep only completion boundaries still needed by unacknowledged arrivals.

    The caller serializes this object with the existing execution lock and
    supplies the local C++ admission-clock watermark. No age/count cutoff is
    used: a delayed or concurrently prepared earlier arrival must remain exact.
    """
    def __init__(self, initial_completed_ns=0):
        self._times = [initial_completed_ns]
        self._checkpoints = [CompletedWorkCheckpoint(initial_completed_ns, NativeWorkPrefix())]
        self._head = 0

    def append(self, completed_ns, prefix, request_refs):
        if type(completed_ns) is not int or completed_ns < self._times[-1]:
            raise ValueError('Native completion clock moved backwards')
        if prefix.step_count != self._checkpoints[-1].prefix.step_count + 1:
            raise ValueError('Native completed work skipped a positive step')
        self._times.append(completed_ns)
        self._checkpoints.append(CompletedWorkCheckpoint(completed_ns, prefix, tuple(request_refs)))

    def capture(self, ready_ns, predecessor_refs=(), pending_refs=()):
        if type(ready_ns) is not int or ready_ns < self._times[self._head]:
            raise ValueError('Native admission boundary was already retired')
        index = bisect_right(self._times, ready_ns, lo=self._head) - 1
        refs = list(predecessor_refs)
        refs.extend(pending_refs)
        # These requests may have finished after C++ receipt but before Python
        # handoff, and therefore disappeared from the live scheduler table.
        for checkpoint in self._checkpoints[index + 1:]:
            refs.extend(checkpoint.request_refs)
        return NativeWorkWindow.capture(self._checkpoints[index].prefix, refs)

    def prune(self, retained_ready_ns):
        if type(retained_ready_ns) is not int or retained_ready_ns < self._times[self._head]:
            raise ValueError('Native admission retention watermark moved backwards')
        head = bisect_right(self._times, retained_ready_ns, lo=self._head) - 1
        for index in range(self._head, head):
            self._checkpoints[index] = None
        self._head = head
        current = self._checkpoints[head]
        # At/before the earliest retained receipt only its integer prefix is
        # needed; completed-request ownership must not leak through this anchor.
        if current.request_refs:
            self._checkpoints[head] = CompletedWorkCheckpoint(current.completed_ns, current.prefix)
        if self._head >= 64 and self._head * 2 >= len(self._times):
            self._times = self._times[self._head:]
            self._checkpoints = self._checkpoints[self._head:]
            self._head = 0

    @property
    def earliest_completed_ns(self):
        return self._times[self._head]

    @property
    def retained_count(self):
        return len(self._times) - self._head
