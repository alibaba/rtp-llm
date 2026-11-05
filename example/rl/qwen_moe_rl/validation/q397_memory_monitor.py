"""NVMLv2-only whole-device memory peaks on one four-GPU node.

No Torch, CUDA runtime, GPU context or model work is created. The primary metric
is nvmlMemory_v2.used. Driver-reserved memory and used+reserved are separate;
these are not PyTorch allocator reserved/allocated counters. Peaks are observed
samples rather than hardware-latched maxima.
"""
from __future__ import annotations
import argparse
import csv
import ctypes
import ctypes.util
import gzip
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import signal
import socket
import time

class MemoryV2(ctypes.Structure):
    _fields_ = [('version', ctypes.c_uint), ('total', ctypes.c_ulonglong),
                ('reserved', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong),
                ('used', ctypes.c_ulonglong)]


class Nvml:
    def __init__(self):
        self.lib = ctypes.CDLL(ctypes.util.find_library('nvidia-ml') or 'libnvidia-ml.so.1')
        self.lib.nvmlErrorString.argtypes = [ctypes.c_int]
        self.lib.nvmlErrorString.restype = ctypes.c_char_p
        self.check(self.lib.nvmlInit_v2())
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.nvmlDeviceGetMemoryInfo_v2.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryV2)]
        self.lib.nvmlDeviceGetUUID.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        count = ctypes.c_uint()
        self.check(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count)))
        if count.value != 4:
            raise RuntimeError(f'Expected four allocated GPUs, found {count.value}')
        self.devices = []
        for index in range(count.value):
            handle = ctypes.c_void_p()
            self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)))
            uuid = ctypes.create_string_buffer(128)
            self.check(self.lib.nvmlDeviceGetUUID(handle, uuid, len(uuid)))
            self.devices.append((index, uuid.value.decode(), handle))

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.nvmlErrorString(code).decode())

    def sample(self):
        result = []
        for index, uuid, handle in self.devices:
            value = MemoryV2()
            value.version = ctypes.sizeof(MemoryV2) | (2 << 24)
            self.check(self.lib.nvmlDeviceGetMemoryInfo_v2(handle, ctypes.byref(value)))
            result.append(dict(index=index, uuid=uuid, query_unix_ns=time.time_ns(),
                               query_monotonic_ns=time.monotonic_ns(),
                               total_bytes=value.total, reserved_bytes=value.reserved,
                               free_bytes=value.free, used_bytes=value.used))
        return result

    def close(self):
        self.check(self.lib.nvmlShutdown())


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as output:
        print(json.dumps(value, indent=2), file=output)
    temporary.replace(path)


class Peaks:
    def __init__(self):
        self.devices = {}
        self.samples = 0
        self.first_unix_ns = self.last_unix_ns = None
        self.last_monotonic_ns = None
        self.maximum_cycle_gap_ms = self.maximum_device_query_span_ms = 0.0
        self.maximum_per_device_query_gap_ms = 0.0
        self.last_device_query_ns = {}
        self.maximum_total_used_bytes = 0
        self.total_peak = None

    def add(self, rows, monotonic_ns):
        if len(rows) != 4 or len({row['uuid'] for row in rows}) != 4:
            raise RuntimeError('A sampling cycle must contain four distinct physical GPUs')
        if self.last_monotonic_ns is not None:
            self.maximum_cycle_gap_ms = max(self.maximum_cycle_gap_ms, (monotonic_ns-self.last_monotonic_ns)/1e6)
        self.last_monotonic_ns = monotonic_ns
        self.samples += 1
        stamps = [row['query_unix_ns'] for row in rows]
        self.first_unix_ns = self.first_unix_ns or min(stamps)
        self.last_unix_ns = max(stamps)
        self.maximum_device_query_span_ms = max(self.maximum_device_query_span_ms, (max(stamps)-min(stamps))/1e6)
        for row in rows:
            prior = self.last_device_query_ns.get(row['index'])
            if prior is not None:
                self.maximum_per_device_query_gap_ms = max(self.maximum_per_device_query_gap_ms,
                    (row['query_monotonic_ns']-prior)/1e6)
            self.last_device_query_ns[row['index']] = row['query_monotonic_ns']
            item = self.devices.setdefault(row['index'], dict(index=row['index'], uuid=row['uuid'],
                total_bytes=row['total_bytes'], baseline_used_bytes=row['used_bytes'],
                baseline_reserved_bytes=row['reserved_bytes'], peak_reserved_bytes=-1,
                peak_used_bytes=-1, peak_used_plus_reserved_bytes=-1))
            if item['uuid'] != row['uuid'] or item['total_bytes'] != row['total_bytes']:
                raise RuntimeError('GPU identity or capacity changed during measurement')
            item['last_used_bytes'] = row['used_bytes']
            item['last_reserved_bytes'] = row['reserved_bytes']
            item['last_free_bytes'] = row['free_bytes']
            if row['reserved_bytes'] > item['peak_reserved_bytes']:
                item.update(peak_reserved_bytes=row['reserved_bytes'],
                            reserved_peak_unix_ns=row['query_unix_ns'])
            if row['used_bytes'] > item['peak_used_bytes']:
                item.update(peak_used_bytes=row['used_bytes'], peak_unix_ns=row['query_unix_ns'],
                            reserved_at_peak_bytes=row['reserved_bytes'])
            physical = row['used_bytes']+row['reserved_bytes']
            if physical > item['peak_used_plus_reserved_bytes']:
                item.update(peak_used_plus_reserved_bytes=physical,
                            physical_peak_unix_ns=row['query_unix_ns'])
        total = sum(row['used_bytes'] for row in rows)
        if total > self.maximum_total_used_bytes:
            self.maximum_total_used_bytes = total
            self.total_peak = dict(unix_ns=max(stamps), query_span_ms=(max(stamps)-min(stamps))/1e6,
                                   per_gpu_used_bytes=[row['used_bytes'] for row in rows])

    def result(self):
        return dict(sample_cycles=self.samples, first_sample_unix_ns=self.first_unix_ns,
                    last_sample_unix_ns=self.last_unix_ns, maximum_cycle_gap_ms=self.maximum_cycle_gap_ms,
                    maximum_device_query_span_ms=self.maximum_device_query_span_ms,
                    maximum_per_device_query_gap_ms=self.maximum_per_device_query_gap_ms,
                    per_gpu=[self.devices[key] for key in sorted(self.devices)],
                    maximum_same_cycle_total_used_bytes=self.maximum_total_used_bytes,
                    total_peak=self.total_peak)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Fresh per-node directory, normally RUN_ROOT/telemetry/NODE_IP')
    parser.add_argument('--node-ip', default=socket.gethostbyname(socket.gethostname()))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--stage', choices=('formal_rl_only','cold_native_reference'), default='formal_rl_only')
    parser.add_argument('--interval-ms', type=float, default=20.)
    parser.add_argument('--probe-only-seconds', type=float, default=0., help='Optional bounded qualification of the monitor')
    args = parser.parse_args()
    ipaddress.ip_address(args.node_ip)
    if not 5 <= args.interval_ms <= 1000 or args.probe_only_seconds < 0:
        parser.error('Interval must be 5–1000ms and probe duration cannot be negative')
    directory = args.output_dir.resolve()
    if directory.exists() and any(directory.iterdir()):
        parser.error('Use a fresh per-node measurement output directory')
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(schema='qwen397-four-gpu-nvml-memory-v1', run_id=args.run_id,
                    hostname=socket.gethostname(), node_ip=args.node_ip, pid=os.getpid(),
                    output_dir=str(directory), interval_ms=args.interval_ms,
                    memory_api='nvmlDeviceGetMemoryInfo_v2', expected_physical_gpus=4,
                    primary_metric='NVML Memory_v2.used, whole physical device across all processes',
                    reserved_metric='NVML Memory_v2.reserved, driver-reserved bytes; not Torch allocator reservation',
                    gpu_context_created=False, stage=args.stage, started_unix_ns=time.time_ns(),
                    started_monotonic_ns=time.monotonic_ns())
    requested_stop = False
    stop_signal = None
    stop_requested_unix_ns = None

    def stop(signum, frame):
        nonlocal requested_stop, stop_signal, stop_requested_unix_ns
        requested_stop = True
        stop_signal = signum
        stop_requested_unix_ns = time.time_ns()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    nvml = None
    peaks = Peaks()
    errors = []
    raw = directory/'gpu-memory.csv.gz'
    next_sample = time.monotonic()
    started = next_sample
    next_checkpoint = next_sample
    deadline_misses = 0
    try:
        nvml = Nvml()
        with gzip.open(raw, 'xt', compresslevel=1, newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample_id','query_unix_ns','query_monotonic_ns','cycle_monotonic_ns',
                             'gpu_index','gpu_uuid','total_bytes','used_bytes','reserved_bytes','free_bytes'])
            while True:
                cycle = time.monotonic_ns()
                rows = nvml.sample()
                peaks.add(rows, cycle)
                writer.writerows([peaks.samples,row['query_unix_ns'],row['query_monotonic_ns'],cycle,
                                  row['index'],row['uuid'],row['total_bytes'],row['used_bytes'],
                                  row['reserved_bytes'],row['free_bytes']] for row in rows)
                now = time.monotonic()
                if peaks.samples == 1:
                    write_json(directory/'ready.json',dict(**identity,status='ready',
                               devices=[dict(index=row['index'],uuid=row['uuid'],total_bytes=row['total_bytes']) for row in rows],
                               first_sample_unix_ns=peaks.first_unix_ns))
                    print('Q397_NVML_MONITOR_READY',json.dumps(dict(pid=os.getpid(),node_ip=args.node_ip,
                          gpu_uuids=[row['uuid'] for row in rows])),flush=True)
                if now >= next_checkpoint:
                    stream.flush()
                    write_json(directory/'live.json',dict(**identity,status='running',**peaks.result()))
                    next_checkpoint = now+2
                if requested_stop or (args.probe_only_seconds and now-started >= args.probe_only_seconds):
                    break
                next_sample += args.interval_ms/1000
                delay = next_sample-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    deadline_misses += 1
                    next_sample = time.monotonic()
    except Exception as error:
        errors.append(type(error).__name__+': '+str(error))
    finally:
        if nvml is not None:
            try:
                nvml.close()
            except Exception as error:
                errors.append('NVML shutdown: '+repr(error))
        result = dict(**identity,status='completed' if not errors else 'failed',
                      passed=not errors and peaks.samples>0,errors=errors,**peaks.result(),
                      deadline_misses=deadline_misses,stop_signal=stop_signal,
                      stop_requested_unix_ns=stop_requested_unix_ns,finished_unix_ns=time.time_ns(),
                      finished_monotonic_ns=time.monotonic_ns(),
                      sampled_peak_limitation='20ms target sampling may miss shorter transient peaks; actual maximum sample gaps are reported.')
        if raw.exists():
            result['raw_samples'] = dict(path=str(raw),bytes=raw.stat().st_size,sha256=file_sha256(raw))
        write_json(directory/'memory-peaks.json',result)
        print('Q397_NVML_MONITOR_FINISHED',json.dumps(dict(passed=result['passed'],node_ip=args.node_ip,
              samples=peaks.samples,maximum_cycle_gap_ms=peaks.maximum_cycle_gap_ms)),flush=True)
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
