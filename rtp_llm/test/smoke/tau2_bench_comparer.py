import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from typing import Any, List, Optional

from rtp_llm.test.smoke.base_comparer import BaseComparer
from rtp_llm.test.smoke.common_def import ABS_PATH, REL_PATH, QueryStatus, SmokeException

TAU2_TARBALL_URL = os.environ.get(
    "TAU2_TARBALL_URL",
    "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/tau2-bench.tar.gz",
)

DEFAULT_THRESHOLD = 0.76
DEFAULT_MODEL_ARG = "Qwen3-30B"
DEFAULT_TASK_IDS_FILE = "passing_tasks.json"
DEFAULT_SCRIPT_FILE = "run_tau2_bench.py"
EVALSCOPE_PINNED_VERSION = "1.6.0"

_REPORT_PATH_RE = re.compile(r"Dump report to:\s*(\S+\.json)")


class Tau2BenchComparer(BaseComparer):
    """Runs tau2-bench against the running smoke server and asserts OVERALL score >= threshold."""

    def run(self):
        out_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd())

        threshold = float(self.qr_info.get("tau2_threshold", DEFAULT_THRESHOLD))
        model_arg = self.qr_info.get("tau2_model", DEFAULT_MODEL_ARG)
        task_ids_file = self.qr_info.get("tau2_task_ids_file", DEFAULT_TASK_IDS_FILE)
        script_file = self.qr_info.get("tau2_script_file", DEFAULT_SCRIPT_FILE)

        extract_root = self._download_and_extract(out_dir)
        self._install_evalscope()
        self._install_tau2_from_tarball(extract_root)
        task_ids_path = self._resolve_task_ids_file(extract_root, task_ids_file)
        script_path = self._resolve_script_file(extract_root, script_file)

        host = "127.0.0.1"
        port = int(self.server_manager.port)
        log_path = os.path.join(out_dir, "tau2_regression.log")
        stdout_text = self._run_tau2_script(
            extract_root, script_path, model_arg, task_ids_path, host, port, log_path
        )

        report_path = self._resolve_report_path(stdout_text, extract_root)
        if report_path is None:
            raise SmokeException(
                QueryStatus.OTHERS,
                f"no 'Dump report to: ...json' line in tau2-bench stdout, see {log_path}",
            )
        score = self._load_overall_score(report_path)
        logging.info(
            f"[TAU2] parsed score={score} from {report_path}, threshold={threshold}"
        )
        if score is None:
            raise SmokeException(
                QueryStatus.OTHERS,
                f"failed to extract OVERALL score from report {report_path}, see {log_path}",
            )
        if score < threshold:
            raise SmokeException(
                QueryStatus.COMPARE_FAILED,
                f"tau2-bench OVERALL score {score} < threshold {threshold}",
            )
        logging.info(f"[TAU2] PASS: OVERALL score={score} >= threshold={threshold}")

    def _pip_install(self, args: List[str]) -> None:
        cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + args
        logging.info(f"[TAU2] pip install: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    def _install_evalscope(self) -> None:
        pinned_spec = f"evalscope=={EVALSCOPE_PINNED_VERSION}"
        try:
            import evalscope

            current = getattr(evalscope, "__version__", None)
            if current == EVALSCOPE_PINNED_VERSION:
                logging.info(
                    f"[TAU2] evalscope=={current} matches pinned, skip install"
                )
                return
            logging.info(
                f"[TAU2] evalscope=={current} != pinned {EVALSCOPE_PINNED_VERSION}, "
                f"force reinstall"
            )
            self._pip_install(["--force-reinstall", "--no-deps", pinned_spec])
            return
        except ImportError:
            pass
        self._pip_install([pinned_spec])

    def _install_tau2_from_tarball(self, extract_root: str) -> None:
        try:
            import tau2  # noqa: F401

            logging.info("[TAU2] tau2 already importable, skip install")
            return
        except ImportError:
            pass

        wheels = sorted(
            glob.glob(os.path.join(extract_root, "**", "*.whl"), recursive=True)
        )
        if wheels:
            logging.info(f"[TAU2] installing tau2 from wheel(s): {wheels}")
            self._pip_install(wheels)
            return

        candidates = [extract_root]
        for sub in os.listdir(extract_root):
            full = os.path.join(extract_root, sub)
            if os.path.isdir(full):
                candidates.append(full)
        for cand in candidates:
            if os.path.isfile(os.path.join(cand, "pyproject.toml")) or os.path.isfile(
                os.path.join(cand, "setup.py")
            ):
                logging.info(f"[TAU2] installing tau2 from source dir: {cand}")
                self._pip_install([cand])
                return

        raise SmokeException(
            QueryStatus.OTHERS,
            f"no installable tau2 package (wheel/setup.py/pyproject.toml) found under {extract_root}",
        )

    def _download_and_extract(self, work_dir: str) -> str:
        dest_dir = os.path.join(work_dir, "tau2-bench")
        if os.path.isdir(dest_dir) and os.listdir(dest_dir):
            logging.info(f"[TAU2] reusing existing extract dir {dest_dir}")
            return self._resolve_extract_root(dest_dir)

        os.makedirs(dest_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tarball_path = tmp.name
        try:
            logging.info(f"[TAU2] downloading tarball to {tarball_path}")
            urllib.request.urlretrieve(TAU2_TARBALL_URL, tarball_path)
            logging.info(f"[TAU2] extracting to {dest_dir}")
            with tarfile.open(tarball_path, "r:gz") as tar:
                tar.extractall(path=dest_dir)
        finally:
            try:
                os.remove(tarball_path)
            except OSError:
                pass
        return self._resolve_extract_root(dest_dir)

    @staticmethod
    def _resolve_extract_root(dest_dir: str) -> str:
        entries = [e for e in os.listdir(dest_dir) if not e.startswith(".")]
        if len(entries) == 1:
            only = os.path.join(dest_dir, entries[0])
            if os.path.isdir(only) and os.path.isdir(os.path.join(only, "scripts")):
                return only
        return dest_dir

    def _resolve_task_ids_file(self, extract_root: str, task_ids_file: str) -> str:
        if os.path.isabs(task_ids_file) and os.path.exists(task_ids_file):
            return task_ids_file

        in_extract = os.path.join(extract_root, task_ids_file)
        if os.path.exists(in_extract):
            return in_extract

        taskinfo_rel_path = self.qr_info.get("_taskinfo_rel_path", "")
        candidates = []
        if taskinfo_rel_path:
            candidates.append(
                os.path.join(os.path.dirname(taskinfo_rel_path), task_ids_file)
            )
        for cand in candidates:
            if os.path.exists(cand):
                dest = os.path.join(extract_root, os.path.basename(task_ids_file))
                shutil.copy(cand, dest)
                logging.info(f"[TAU2] copied {cand} -> {dest}")
                return dest

        raise SmokeException(
            QueryStatus.OTHERS,
            f"task_ids_file {task_ids_file} not found; tried {[in_extract] + candidates}",
        )

    def _resolve_script_file(self, extract_root: str, script_file: str) -> str:
        if os.path.isabs(script_file) and os.path.isfile(script_file):
            return script_file

        taskinfo_rel_path = self.qr_info.get("_taskinfo_rel_path", "")
        candidates = []
        if taskinfo_rel_path:
            candidates.append(
                os.path.join(os.path.dirname(taskinfo_rel_path), script_file)
            )
        candidates.append(os.path.join(extract_root, "scripts", script_file))
        candidates.append(os.path.join(extract_root, script_file))

        for cand in candidates:
            if os.path.isfile(cand):
                return os.path.abspath(cand)

        raise SmokeException(
            QueryStatus.OTHERS,
            f"script not found: {script_file}; tried {candidates}",
        )

    def _run_tau2_script(
        self,
        extract_root: str,
        script_path: str,
        model_arg: str,
        task_ids_path: str,
        host: str,
        port: int,
        log_path: str,
    ) -> str:

        cmd = [
            sys.executable,
            "-u",
            script_path,
            "--model",
            model_arg,
            "--host",
            host,
            "--port",
            str(port),
            "--task-ids-file",
            task_ids_path,
        ]
        logging.info(f"[TAU2] running: {' '.join(cmd)} (cwd={extract_root})")

        captured: list = []
        with open(log_path, "w", encoding="utf-8") as log_fp:
            proc = subprocess.Popen(
                cmd,
                cwd=extract_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log_fp.write(line)
                captured.append(line)
            rc = proc.wait()

        if rc != 0:
            raise SmokeException(
                QueryStatus.OTHERS,
                f"run_tau2_bench.py exited with code {rc}, see {log_path}",
            )
        return "".join(captured)

    @staticmethod
    def _resolve_report_path(stdout_text: str, cwd: str) -> Optional[str]:
        """Find the last 'Dump report to: <path>.json' line and return absolute path."""
        matches = _REPORT_PATH_RE.findall(stdout_text)
        if not matches:
            return None
        rel_path = matches[-1].strip()
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.normpath(os.path.join(cwd, rel_path))

    @staticmethod
    def _load_overall_score(report_path: str) -> Optional[float]:
        """Read evalscope JSON report and return OVERALL mean_acc score."""
        try:
            with open(report_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except (OSError, json.JSONDecodeError) as e:
            logging.warning(f"[TAU2] failed to read report {report_path}: {e}")
            return None

        score, metrics = Tau2BenchComparer._normalize_report_schema(data)
        if score is not None:
            return score
        for metric in metrics:
            if metric["name"] == "mean_acc" and metric["is_overall"]:
                return metric["score"]
        return None

    @staticmethod
    def _normalize_report_schema(data: dict) -> tuple[Optional[float], list[dict[str, Any]]]:
        score = Tau2BenchComparer._to_float(data.get("score"))
        metrics = []
        for raw_metric in data.get("metrics", []) or []:
            if not isinstance(raw_metric, dict):
                continue
            name = Tau2BenchComparer._first_text(
                raw_metric, ["name", "metric", "metric_name"]
            )
            metric_score = Tau2BenchComparer._to_float(
                raw_metric.get("score", raw_metric.get("value"))
            )
            if name is None or metric_score is None:
                continue
            metrics.append(
                {
                    "name": name.lower(),
                    "score": metric_score,
                    "is_overall": Tau2BenchComparer._is_overall_metric(raw_metric),
                }
            )
        return score, metrics

    @staticmethod
    def _is_overall_metric(metric: dict[str, Any]) -> bool:
        if metric.get("overall") is True or metric.get("is_overall") is True:
            return True
        normalized_dimension = Tau2BenchComparer._first_text(
            metric,
            [
                "dimension",
                "dim",
                "scope",
                "group",
                "category",
                "dataset",
                "dataset_name",
                "subset",
                "subset_name",
                "task",
                "task_name",
            ],
        )
        return normalized_dimension == "OVERALL"

    @staticmethod
    def _first_text(data: dict[str, Any], keys: list[str]) -> Optional[str]:
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()
        return None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        return None
