"""Shared runtime fakes and accounting inputs; no collected tests."""

import copy
import threading
from types import SimpleNamespace as NS

from program_fixtures import compile_scenarios


def source():
    return {
        "implementation": {},
        "id": "lifecycle",
        "description": "Request lifecycle",
        "category": "status",
        "profiles": ["batch-window"],
        "environment": {},
        "execution": {"timeout_s": 3, "stage_timeout_s": 2, "cleanup_timeout_s": 5},
        "stages": [
            {"id": "setup", "action": "setup"},
            {"id": "submit", "action": "request"},
            {"id": "wait", "action": "wait", "params": {"requests": {"$ref": "stages.submit.output.requests"}}},
            {
                "id": "check",
                "action": "check",
                "params": {"actual": {"$ref": "stages.wait.output.completed"}, "op": "eq", "expected": True},
            },
        ],
    }


class Clock:
    now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Backend:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.fail_setup = self.fail_submit = self.fail_cleanup = self.timeout_wait = False
        self.completed = True

    def setup(self, ctx, environment, deadline):
        self.calls.append("setup")
        if self.fail_setup:
            raise RuntimeError("partial setup")
        return object(), object()

    def start_requests(self, ctx, params, deadline):
        self.calls.append("submit")
        records = [1]

        def cleanup(d):
            self.calls.append("request_cleanup")
            d.sleep(0.5)
            if self.fail_cleanup:
                raise RuntimeError("cancel failed")

        handle = ctx.register_resource("requests", records, cleanup)
        if self.fail_submit:
            raise RuntimeError("partial submit")
        return handle

    def wait_requests(self, ctx, resource, deadline):
        self.calls.append("wait")
        if self.timeout_wait:
            deadline.sleep(4)
        return {"completed": self.completed, "error_count": int(not self.completed)}

    def cancel_requests(self, ctx, resource, deadline):
        return len(resource)

    def teardown(self, ctx, deadline):
        self.calls.append("teardown")
        deadline.sleep(0.5)


class Stream:
    def __init__(self, hanging=False, error=None):
        self.hanging, self.error = hanging, error
        self.cancelled = threading.Event()

    def cancel(self):
        self.cancelled.set()
        return True

    def __iter__(self):
        if self.hanging:
            self.cancelled.wait(2)
            if not self.cancelled.is_set():
                raise RuntimeError("test stream cancellation was not issued")
            return
        if self.error:
            raise self.error
        yield NS(HasField=lambda key: False, flatten_output=NS(finished=[True]))


class Ops:
    def __init__(self, batch=True, hanging=False):
        self.batch, self.hanging = batch, hanging
        self.streams, self.fetch_count, self.generate_count, self.counter = [], 0, 0, 0
        self.responses = []
        self.schedule_pb2_grpc = NS(FlexlbServiceStub=lambda channel: NS(Schedule=NS(future=self.future)))
        self.pb2_grpc = NS(RpcServiceStub=lambda channel: NS(FetchResponse=self.fetch, GenerateStreamCall=self.generate))
        self.pb2 = NS(FetchRequestPB=lambda **kw: kw)

    def future(self, req, timeout, metadata=None):
        self.last_schedule = (req, timeout, metadata)
        response = NS(code=200, success=True, error_message="", enqueued_by_master=self.batch)
        self.responses.append(response)
        return NS(result=lambda timeout: response, cancel=lambda: True)

    def next_request_id(self):
        self.counter += 1
        return self.counter

    def _channel(self, target):
        return target

    def master_target(self):
        return "master"

    def build_schedule_request(self, rid, **shape):
        return (rid, shape)

    def prefill_addr(self, response):
        return "prefill"

    def build_generate_input(self, rid, **shape):
        return (rid, shape)

    def _copy_role_addrs(self, inp, response):
        pass

    def fetch(self, req, timeout):
        self.fetch_count += 1
        stream = Stream(self.hanging)
        self.streams.append(stream)
        return stream

    def generate(self, req, timeout):
        self.generate_count += 1
        stream = Stream(self.hanging)
        self.streams.append(stream)
        return stream


def lease_manifest():
    m, b = 28000, 55000
    env = dict(
        FLEXLB_FT_MASTER_HTTP_PORT=str(m),
        FLEXLB_FT_MASTER_MANAGEMENT_PORT=str(m + 1),
        FLEXLB_FT_HA_MASTER_A_HTTP_PORT=str(m),
        FLEXLB_FT_HA_MASTER_B_HTTP_PORT=str(m + 3),
        FLEXLB_FT_MOCK_BASE_GRPC_PORT=str(b),
    )
    return dict(
        lease_schema_version=1,
        lane=0,
        backend="java_mock",
        master_base=m,
        mock_base=b,
        worker_capacity=6,
        intervals=[dict(side="master", first=m, last=m + 5), dict(side="mock", first=b - 1, last=b + 151)],
        lock_names=[f"m{m}_{m + 5}.lock", f"g{b - 1}_{b + 151}.lock"],
        child_env=env,
    )


def catalog():
    document = {
        "instance_catalog_schema_version": 1,
        "instances": [
            {
                "id": f"scenario{i}::default::batch-window",
                "scenario_id": f"scenario{i}",
                "variant_id": "default",
                "profile": "batch-window",
                "category": "cancel" if i % 2 else "kv",
                "source_path": f"config/scenarios/scenario{i}.yaml",
                "source": "yaml",
                "tags": [],
                "requires": [],
                "estimated_duration_s": i + 1,
                "metadata": {"kind": "functional", "description": "Fixture"},
                "execution": {
                    "timeout_s": 60,
                    "stage_timeout_s": 30,
                    "cleanup_timeout_s": 10,
                    "finalize_timeout_s": 20,
                    "report_timeout_s": 10,
                    "archive_timeout_s": 10,
                    "collection": "diagnostic",
                },
                "resource_budget": {
                    "backend": "java_mock",
                    "initial_workers": 6,
                    "max_dynamic_additions": 0,
                    "bounded": True,
                    "mock_control_offset": -1,
                    "victim_control_offset": 149,
                    "victim_grpc_offset": 150,
                    "reserved_tail_offset": 151,
                },
            }
            for i in range(7)
        ],
    }

    for row in document["instances"]:
        declaration = source()
        declaration.update(id=row["scenario_id"], category=row["category"], description=row["metadata"]["description"])
        compiled = compile_scenarios([(row["source_path"], declaration)])[0]
        compiled.update(copy.deepcopy(row))
        row["compiled"] = compiled
    return document
