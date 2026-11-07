"""Run a fixed operator DAG on one stream or across streams joined by events.

Each node names its stream and the nodes it depends on. In multi-stream mode a node waits for every
producer on another stream, and each side stream rejoins the caller's stream at the end, so the DAG
is a single fork/join region that CUDA Graph capture accepts. In single-stream mode the same nodes run
in the order they were added, which must therefore be a topological order.

Multi-stream mode captures the nodes in its own order. CUDA Graph replay assigns its own streams, and
a node with parents on several streams has been observed to continue on the stream of the parent
captured first; a main-stream node whose side-stream parent came first moves the rest of the main
chain onto another replay stream, and repeated layers then spread over dozens of streams. So each
side-stream node is captured just before the main-stream node that first needs it (after that node's
main-stream predecessor), and the final join skips side streams that already reach the last
main-stream node. Dependencies are unchanged; only capture order and redundant join edges differ.

Side streams are created by the constructor and events by finalize(), both before any capture; run()
only records and waits.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import torch

MAIN = "main"


@dataclass(frozen=True)
class _Node:
    name: str
    stream: str
    after: Tuple[str, ...]
    fn: Callable[[Any], None]


class StreamDag:
    def __init__(self, device, side_streams=()):
        self._streams = {
            name: torch.cuda.Stream(device=device) for name in side_streams
        }
        self._nodes = []
        self._events = None

    def add(self, name, fn, *, stream=MAIN, after=()):
        known = {node.name for node in self._nodes}
        if name in known:
            raise ValueError(f"duplicate DAG node {name}")
        missing = [dep for dep in after if dep not in known]
        if missing:
            raise ValueError(f"{name} depends on nodes not added before it: {missing}")
        if stream != MAIN and stream not in self._streams:
            raise ValueError(f"unknown stream {stream}")
        self._nodes.append(_Node(name, stream, tuple(after), fn))
        self._events = None

    def _capture_order(self) -> List[_Node]:
        """Main-stream nodes in order; side-stream ancestors of each one just before it."""
        by_name = {node.name: node for node in self._nodes}
        previous: Dict[str, str] = {}  # node -> preceding node on its stream
        last: Dict[str, str] = {}
        for node in self._nodes:
            if node.stream in last:
                previous[node.name] = last[node.stream]
            last[node.stream] = node.name
        order: List[_Node] = []
        emitted = set()

        def emit(name):
            if name in emitted:
                return
            node = by_name[name]
            for parent in (*node.after, previous.get(name)):
                if parent is not None:
                    emit(parent)
            emitted.add(name)
            order.append(node)

        for node in self._nodes:
            if node.stream != MAIN:
                continue
            if name := previous.get(node.name):
                emit(name)
            emit(node.name)
        for node in self._nodes:
            emit(node.name)
        return order

    def finalize(self):
        """Create the events (one per cross-stream producer and per side-stream tail that the main
        stream does not already reach) and fix the capture order."""
        streams = {node.name: node.stream for node in self._nodes}
        needed = {
            dep
            for node in self._nodes
            for dep in node.after
            if streams[dep] != node.stream
        }
        tails: Dict[str, str] = {}
        for node in self._nodes:
            if node.stream != MAIN:
                tails[node.stream] = node.name
        reached = set()
        main_nodes = [node.name for node in self._nodes if node.stream == MAIN]
        if main_nodes:
            parents = {node.name: list(node.after) for node in self._nodes}
            previous = None
            for node in self._nodes:  # stream order is a dependency too
                if node.stream == MAIN:
                    if previous is not None:
                        parents[node.name].append(previous)
                    previous = node.name
            for stream in self._streams:
                chain = [node.name for node in self._nodes if node.stream == stream]
                for before, after in zip(chain, chain[1:]):
                    parents[after].append(before)
            stack = [main_nodes[-1]]
            while stack:
                name = stack.pop()
                if name not in reached:
                    reached.add(name)
                    stack.extend(parents[name])
        self._tails = {s: t for s, t in tails.items() if t not in reached}
        needed.update(self._tails.values())
        self._streams_of = streams
        self._order = self._capture_order()
        self._events = {name: torch.cuda.Event() for name in needed}
        self._fork = torch.cuda.Event()
        # Create the event handles outside capture.
        for event in (*self._events.values(), self._fork):
            event.record()
        torch.cuda.current_stream().synchronize()

    def run(self, state, *, multi_stream=True):
        if self._events is None:
            raise RuntimeError("StreamDag.finalize() must run before run()")
        if not multi_stream:
            for node in self._nodes:
                node.fn(state)
            return
        main = torch.cuda.current_stream()
        self._fork.record(main)
        forked = set()
        for node in self._order:
            stream = main if node.stream == MAIN else self._streams[node.stream]
            with torch.cuda.stream(stream):
                if node.stream != MAIN and node.stream not in forked:
                    stream.wait_event(self._fork)
                    forked.add(node.stream)
                for dep in node.after:
                    if self._streams_of[dep] != node.stream:
                        stream.wait_event(self._events[dep])
                node.fn(state)
                event = self._events.get(node.name)
                if event is not None:
                    event.record(stream)
        for tail in self._tails.values():
            main.wait_event(self._events[tail])
