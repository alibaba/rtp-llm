"""Stage DSV4.1 HTTP inputs on the engine's shared download pool."""

import mmap
import os
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from multiprocessing.util import get_temp_dir

from rtp_llm.multimodal.mm_preprocess import (
    _report_vit_preprocess_samples,
    download_region,
)
from rtp_llm.multimodal.multimodal_util import collect_download_timing
from rtp_llm.multimodal.vit_metrics import collect_vit_preprocess_metrics
from rtp_llm.telemetry import vit_media


@dataclass(frozen=True)
class FileSource:
    """Only the path is sent through the existing CPU task queue."""

    path: str

    @contextmanager
    def open(self):
        with open(self.path, "rb") as file:
            if not os.fstat(file.fileno()).st_size:
                yield file
                return
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
                yield data


class _Download:
    def __init__(self):
        self.files = ExitStack()
        self.started = self.finished = 0.0
        self.network_intervals = []
        self.future = None

    def close(self):
        if self.future is not None:
            self.future.cancel()
            # Running HTTP reads still own the source until they return.
            files = self.files
            self.future.add_done_callback(lambda _: files.close())


class _Downloads:
    def __init__(self, count):
        self.cancelled = threading.Event()
        self.jobs = [None] * count

    def ready(self):
        for job in self.jobs:
            if job is not None and job.future.done():
                # Propagate failed media even while siblings are still downloading.
                job.future.result()
        return all(job is None or job.future.done() for job in self.jobs)

    def result(self):
        return tuple(job.future.result() if job else None for job in self.jobs)

    @property
    def elapsed_ms(self):
        jobs = [job for job in self.jobs if job is not None]
        return 1000.0 * (
            max(job.finished for job in jobs) - min(job.started for job in jobs)
        )

    @property
    def network_ms(self):
        # Concurrent network attempts contribute wall time only once.
        end = elapsed = 0.0
        for start, stop in sorted(
            interval for job in self.jobs if job for interval in job.network_intervals
        ):
            elapsed += max(0.0, stop - max(start, end))
            end = max(end, stop)
        return elapsed * 1000.0

    def close(self):
        self.cancelled.set()
        for job in self.jobs:
            if job is not None:
                job.close()


class ImageSourcePrefetch:
    def __init__(self, executor, fetch, config):
        self.executor = executor
        self.fetch = fetch
        self.config = config
        # This private directory is owned by the parent process. File-backed
        # pages can be reclaimed when completed downloads queue behind decoding.
        self.directory = None if config.use_local_preprocess else get_temp_dir()

    def start(self, item):
        if not item.should_preprocess:
            return False
        batch = None
        for index, media in enumerate(item.mm_inputs):
            if not media.url.startswith(("http://", "https://")):
                continue
            if batch is None:
                deadline = time.monotonic() + item.remaining_timeout_s()
                item.source_download = batch = _Downloads(len(item.mm_inputs))
            job = batch.jobs[index] = _Download()
            job.future = self.executor.submit(
                self._download,
                job,
                media.url,
                item.media_traces[index : index + 1],
                item.region,
                deadline,
                batch.cancelled,
            )
            ready = item.preprocess_ready_event
            job.future.add_done_callback(lambda _, event=ready: event.set())
        return batch is not None

    def _download(self, job, url, media_traces, region, deadline, cancelled):
        job.started = time.monotonic()
        try:
            with download_region(region), vit_media.bind(
                media_traces
            ), collect_download_timing(
                job.network_intervals
            ), collect_vit_preprocess_metrics() as metrics:
                try:
                    if self.directory is None:
                        source = self.fetch(url, self.config, deadline, cancelled, None)
                        return job.files.enter_context(source)
                    # A unique, closed-after-write file avoids an fd per queued
                    # image and opens directly in spawned children. No second pipe.
                    with tempfile.NamedTemporaryFile(
                        dir=self.directory, delete=False
                    ) as file:
                        job.files.callback(os.unlink, file.name)
                        self.fetch(url, self.config, deadline, cancelled, file)
                        return FileSource(file.name)
                finally:
                    job.finished = time.monotonic()
                    _report_vit_preprocess_samples(metrics.samples)
        except BaseException:
            job.files.close()
            raise
