import ctypes
import functools
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import inspect
except Exception:  # pragma: no cover
    inspect = None


_LOGGER = logging.getLogger(__name__)

_ENV_ENABLE = "RTP_HOT_HOOK"
_ENV_HOOK_FILE = "RTP_HOT_HOOK_FILE"
_ENV_CONFIG = "RTP_HOT_HOOK_CONFIG"
_ENV_DUMP_DIR = "RTP_HOT_HOOK_DUMP_DIR"
_ENV_ALLOW_LOCAL_MUTATION = "RTP_HOT_HOOK_ALLOW_LOCAL_MUTATION"


def _truthy(value: Optional[str]) -> bool:
    return value is not None and value.lower() in ("1", "true", "yes", "on")


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def _json_default(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return value
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    if shape is not None and dtype is not None:
        return {
            "type": type(value).__name__,
            "shape": list(shape),
            "dtype": str(dtype),
            "device": str(device),
        }
    return repr(value)


def _copy_to_cpu(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu()
    return value


def _frame_locals_to_fast(frame: FrameType) -> None:
    try:
        ctypes.pythonapi.PyFrame_LocalsToFast.argtypes = [
            ctypes.py_object,
            ctypes.c_int,
        ]
        ctypes.pythonapi.PyFrame_LocalsToFast.restype = None
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))
    except Exception as e:
        _LOGGER.warning("failed to write frame locals back: %s", e)


@dataclass
class FunctionHook:
    target: str
    enabled: bool = True
    before: List[str] = field(default_factory=list)
    after: List[str] = field(default_factory=list)
    exception: List[str] = field(default_factory=list)
    replace: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineHook:
    file: Optional[str]
    line: int
    enabled: bool = True
    hook: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    file_suffix: Optional[str] = None


@dataclass
class _PatchedTarget:
    parent: Any
    attr: str
    original_descriptor: Any
    original_callable: Callable[..., Any]
    descriptor_kind: str


class HookContext:
    def __init__(
        self,
        runtime: "HotHookRuntime",
        kind: str,
        target: str,
        event: str,
        hook_config: Dict[str, Any],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        result: Any = None,
        exception: Optional[BaseException] = None,
        frame: Optional[FrameType] = None,
    ):
        self.runtime = runtime
        self.kind = kind
        self.target = target
        self.event = event
        self.hook_config = hook_config
        self.config = runtime.config
        self.case = runtime.case
        self.args = list(args)
        self.kwargs = dict(kwargs or {})
        self.result = result
        self.exception = exception
        self.frame = frame
        self.timestamp = time.time()

    @property
    def locals(self) -> Dict[str, Any]:
        if self.frame is None:
            return {}
        return self.frame.f_locals

    def set_local(self, name: str, value: Any) -> None:
        if not _truthy(os.environ.get(_ENV_ALLOW_LOCAL_MUTATION)):
            raise RuntimeError(
                f"{_ENV_ALLOW_LOCAL_MUTATION}=1 is required to mutate frame locals"
            )
        if self.frame is None:
            raise RuntimeError("set_local is only available for line hooks")
        self.frame.f_locals[name] = value
        _frame_locals_to_fast(self.frame)

    def dump_path(self, name: str, suffix: str) -> str:
        index = self.runtime.next_dump_index()
        root = Path(self.runtime.dump_dir) / _safe_name(self.case)
        root.mkdir(parents=True, exist_ok=True)
        base = f"{index:06d}_{_safe_name(self.target)}_{_safe_name(name)}{suffix}"
        return str(root / base)

    def dump(self, name: str, value: Any) -> str:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                path = self.dump_path(name, ".pt")
                torch.save(_copy_to_cpu(value), path)
                return path
        except Exception:
            pass

        path = self.dump_path(name, ".json")
        with open(path, "w") as f:
            json.dump(_json_default(value), f, default=_json_default, indent=2)
        return path

    def hash(self, value: Any) -> str:
        value = _copy_to_cpu(value)
        if hasattr(value, "numpy"):
            payload = value.numpy().tobytes()
        else:
            payload = repr(value).encode("utf-8", errors="replace")
        return hashlib.sha256(payload).hexdigest()

    def stats(self, value: Any) -> Dict[str, Any]:
        stats: Dict[str, Any] = _json_default(value)
        if not isinstance(stats, dict):
            stats = {"value": stats}
        try:
            tensor = _copy_to_cpu(value)
            if hasattr(tensor, "float"):
                as_float = tensor.float()
                stats.update(
                    {
                        "min": float(as_float.min().item()),
                        "max": float(as_float.max().item()),
                        "mean": float(as_float.mean().item()),
                    }
                )
        except Exception as e:
            stats["stats_error"] = str(e)
        return stats

    def note(self, name: str, value: Any = None) -> str:
        root = Path(self.runtime.dump_dir) / _safe_name(self.case)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "notes.jsonl"
        record = {
            "time": self.timestamp,
            "kind": self.kind,
            "target": self.target,
            "event": self.event,
            "name": name,
            "value": _json_default(value),
        }
        with open(path, "a") as f:
            f.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")
        return str(path)


class HotHookRuntime:
    def __init__(self) -> None:
        self.enabled = False
        self.config: Dict[str, Any] = {}
        self.case = "default"
        self.dump_dir = ".tmp/rtp_hot_hook"
        self._hook_module: Optional[ModuleType] = None
        self._hook_file: Optional[str] = None
        self._config_file: Optional[str] = None
        self._last_hook_mtime: Optional[float] = None
        self._last_config_mtime: Optional[float] = None
        self._patched: Dict[str, _PatchedTarget] = {}
        self._function_hooks: Dict[str, FunctionHook] = {}
        self._line_hooks: Dict[int, List[LineHook]] = {}
        self._dump_index = 0
        self._installed_trace = False
        self._reload_error: Optional[str] = None
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            self._unpatch_all()
            if self._installed_trace:
                sys.settrace(None)
                threading.settrace(None)
            self.__init__()

    def next_dump_index(self) -> int:
        with self._lock:
            self._dump_index += 1
            return self._dump_index

    def install_if_enabled(self) -> bool:
        if not _truthy(os.environ.get(_ENV_ENABLE)):
            return False
        with self._lock:
            self.enabled = True
            self._hook_file = os.environ.get(_ENV_HOOK_FILE)
            self._config_file = os.environ.get(_ENV_CONFIG)
            self.dump_dir = os.environ.get(_ENV_DUMP_DIR, self.dump_dir)
            if not self.reload(force=True):
                return False
            self._update_trace_state()
            return True

    def reload(self, force: bool = False) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            hook_file = os.environ.get(_ENV_HOOK_FILE, self._hook_file)
            config_file = os.environ.get(_ENV_CONFIG, self._config_file)
            dump_dir = os.environ.get(_ENV_DUMP_DIR, self.dump_dir)
            self._hook_file = hook_file
            self._config_file = config_file
            self.dump_dir = dump_dir

            hook_mtime = self._mtime(hook_file)
            config_mtime = self._mtime(config_file)
            if (
                not force
                and hook_mtime == self._last_hook_mtime
                and config_mtime == self._last_config_mtime
            ):
                return True

            try:
                new_config = self._load_config(config_file)
                if not new_config.get("enabled", True):
                    self.config = new_config
                    self.case = str(new_config.get("case", "default"))
                    self._function_hooks = {}
                    self._line_hooks = {}
                    self._unpatch_all()
                    self._update_trace_state()
                    self._last_config_mtime = config_mtime
                    self._last_hook_mtime = hook_mtime
                    return True

                if not hook_file:
                    raise RuntimeError(f"{_ENV_HOOK_FILE} is required when hooks are enabled")
                new_hook_module = self._load_hook_module(hook_file)
                new_function_hooks = self._parse_function_hooks(new_config)
                new_line_hooks = self._parse_line_hooks(new_config)

                self._validate_callbacks(new_hook_module, new_function_hooks, new_line_hooks)
                self.config = new_config
                self.case = str(new_config.get("case", "default"))
                self._hook_module = new_hook_module
                self._function_hooks = new_function_hooks
                self._line_hooks = new_line_hooks
                self._apply_function_patches()
                self._update_trace_state()
                self._last_hook_mtime = hook_mtime
                self._last_config_mtime = config_mtime
                self._reload_error = None
                return True
            except Exception as e:
                self._reload_error = f"{e}\n{traceback.format_exc()}"
                _LOGGER.error("hot hook reload failed; keeping previous hooks: %s", e)
                return False

    def _mtime(self, path: Optional[str]) -> Optional[int]:
        if not path:
            return None
        try:
            return os.stat(path).st_mtime_ns
        except OSError:
            return None

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        if not config_file:
            return {"enabled": True}
        with open(config_file, "r") as f:
            return json.load(f)

    def _load_hook_module(self, hook_file: str) -> ModuleType:
        module_name = f"_rtp_hot_hook_user_{os.getpid()}"
        module = ModuleType(module_name)
        module.__file__ = hook_file
        with open(hook_file, "r") as f:
            source = f.read()
        code = compile(source, hook_file, "exec")
        exec(code, module.__dict__)
        return module

    def _parse_function_hooks(self, config: Dict[str, Any]) -> Dict[str, FunctionHook]:
        hooks: Dict[str, FunctionHook] = {}
        for raw in config.get("function_hooks", []):
            if not raw.get("enabled", True):
                continue
            target = raw["target"]
            hook = FunctionHook(
                target=target,
                enabled=True,
                before=self._as_list(raw.get("before")),
                after=self._as_list(raw.get("after")),
                exception=self._as_list(raw.get("exception")),
                replace=raw.get("replace"),
                config=raw,
            )
            hooks[target] = hook
        return hooks

    def _parse_line_hooks(self, config: Dict[str, Any]) -> Dict[int, List[LineHook]]:
        hooks: Dict[int, List[LineHook]] = {}
        for raw in config.get("line_hooks", []):
            if not raw.get("enabled", True):
                continue
            file_name = os.path.realpath(raw["file"]) if raw.get("file") else None
            file_suffix = raw.get("file_suffix")
            if file_suffix is not None:
                file_suffix = self._normalize_file_suffix(str(file_suffix))
            if file_name is None and file_suffix is None:
                raise RuntimeError("line hook requires either 'file' or 'file_suffix'")
            hook = LineHook(
                file=file_name,
                line=int(raw["line"]),
                enabled=True,
                hook=self._as_list(raw.get("hook")),
                config=raw,
                file_suffix=file_suffix,
            )
            hooks.setdefault(hook.line, []).append(hook)
        return hooks

    def _normalize_file_suffix(self, suffix: str) -> str:
        suffix = suffix.replace("\\", os.sep).replace("/", os.sep)
        return suffix.lstrip(os.sep)

    def _as_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _validate_callbacks(
        self,
        module: ModuleType,
        function_hooks: Dict[str, FunctionHook],
        line_hooks: Dict[int, List[LineHook]],
    ) -> None:
        names: List[str] = []
        for hook in function_hooks.values():
            names.extend(hook.before)
            names.extend(hook.after)
            names.extend(hook.exception)
            if hook.replace:
                names.append(hook.replace)
        for hook_list in line_hooks.values():
            for hook in hook_list:
                names.extend(hook.hook)
        missing = [name for name in names if not callable(getattr(module, name, None))]
        if missing:
            raise RuntimeError(f"missing hot hook callback(s): {missing}")

    def _update_trace_state(self) -> None:
        should_trace = self.enabled and bool(self._line_hooks)
        if should_trace and not self._installed_trace:
            sys.settrace(self._trace)
            threading.settrace(self._trace)
            self._installed_trace = True
        elif not should_trace and self._installed_trace:
            sys.settrace(None)
            threading.settrace(None)
            self._installed_trace = False

    def _callback(self, name: str) -> Callable[[HookContext], Any]:
        if self._hook_module is None:
            raise RuntimeError("hot hook module is not loaded")
        callback = getattr(self._hook_module, name)
        if not callable(callback):
            raise RuntimeError(f"hot hook callback is not callable: {name}")
        return callback

    def _run_callback(self, name: str, ctx: HookContext) -> Any:
        try:
            return self._callback(name)(ctx)
        except Exception as e:
            _LOGGER.error(
                "hot hook callback failed: %s target=%s event=%s error=%s",
                name,
                ctx.target,
                ctx.event,
                e,
            )
            return None

    def _apply_function_patches(self) -> None:
        for target in list(self._patched):
            if target not in self._function_hooks:
                self._unpatch(target)
        for target in self._function_hooks:
            if target not in self._patched:
                self._patch(target)

    def _patch(self, target: str) -> None:
        parent, attr = self._resolve_parent(target)
        original_descriptor = (
            inspect.getattr_static(parent, attr) if inspect is not None else getattr(parent, attr)
        )
        descriptor_kind = "plain"
        if isinstance(original_descriptor, staticmethod):
            original_callable = original_descriptor.__func__
            descriptor_kind = "staticmethod"
        elif isinstance(original_descriptor, classmethod):
            original_callable = original_descriptor.__func__
            descriptor_kind = "classmethod"
        else:
            original_callable = getattr(parent, attr)

        @functools.wraps(original_callable)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._call_function_hook(target, original_callable, args, kwargs)

        if descriptor_kind == "staticmethod":
            patched_value = staticmethod(wrapper)
        elif descriptor_kind == "classmethod":
            patched_value = classmethod(wrapper)
        else:
            patched_value = wrapper
        setattr(parent, attr, patched_value)
        self._patched[target] = _PatchedTarget(
            parent=parent,
            attr=attr,
            original_descriptor=original_descriptor,
            original_callable=original_callable,
            descriptor_kind=descriptor_kind,
        )
        _LOGGER.info("hot hook patched %s", target)

    def _unpatch(self, target: str) -> None:
        patched = self._patched.pop(target, None)
        if patched is not None:
            setattr(patched.parent, patched.attr, patched.original_descriptor)
            _LOGGER.info("hot hook unpatched %s", target)

    def _unpatch_all(self) -> None:
        for target in list(self._patched):
            self._unpatch(target)

    def _resolve_parent(self, target: str) -> Tuple[Any, str]:
        parts = target.split(".")
        last_error: Optional[Exception] = None
        for idx in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:idx])
            try:
                obj: Any = importlib.import_module(module_name)
            except Exception as e:
                last_error = e
                continue
            for part in parts[idx:-1]:
                obj = getattr(obj, part)
            return obj, parts[-1]
        raise RuntimeError(f"cannot resolve hot hook target {target}: {last_error}")

    def _call_function_hook(
        self,
        target: str,
        original: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        self.reload(force=False)
        hook = self._function_hooks.get(target)
        if hook is None or not hook.enabled:
            return original(*args, **kwargs)
        if hook.replace:
            ctx = HookContext(self, "function", target, "replace", hook.config, args, kwargs)
            replacement = self._run_callback(hook.replace, ctx)
            if replacement is not None:
                return replacement
            return original(*args, **kwargs)

        ctx = HookContext(self, "function", target, "before", hook.config, args, kwargs)
        for name in hook.before:
            self._run_callback(name, ctx)
        try:
            result = original(*tuple(ctx.args), **ctx.kwargs)
        except BaseException as e:
            exc_ctx = HookContext(
                self,
                "function",
                target,
                "exception",
                hook.config,
                tuple(ctx.args),
                ctx.kwargs,
                exception=e,
            )
            for name in hook.exception:
                replacement = self._run_callback(name, exc_ctx)
                if replacement is not None:
                    return replacement
            raise
        after_ctx = HookContext(
            self,
            "function",
            target,
            "after",
            hook.config,
            tuple(ctx.args),
            ctx.kwargs,
            result=result,
        )
        for name in hook.after:
            replacement = self._run_callback(name, after_ctx)
            if replacement is not None:
                result = replacement
                after_ctx.result = result
        return result

    def _trace(self, frame: FrameType, event: str, arg: Any) -> Any:
        if event != "line" or not self.enabled:
            return self._trace
        self.reload(force=False)
        file_name = os.path.realpath(frame.f_code.co_filename)
        line_no = frame.f_lineno
        line_hooks = self._matching_line_hooks(file_name, line_no)
        if not line_hooks:
            return self._trace
        for hook in self._matching_line_hooks(file_name, line_no):
            ctx = HookContext(
                self,
                "line",
                f"{file_name}:{line_no}",
                "line",
                hook.config,
                frame=frame,
            )
            for name in hook.hook:
                result = self._run_callback(name, ctx)
                if isinstance(result, dict):
                    for local_name, local_value in result.items():
                        try:
                            ctx.set_local(local_name, local_value)
                        except Exception as e:
                            _LOGGER.error(
                                "hot hook local mutation failed: %s target=%s error=%s",
                                local_name,
                                ctx.target,
                                e,
                            )
        return self._trace

    def _matching_line_hooks(self, file_name: str, line_no: int) -> List[LineHook]:
        hooks = self._line_hooks.get(line_no, [])
        if not hooks:
            return []
        normalized_file = self._normalize_file_suffix(file_name)
        matched = []
        for hook in hooks:
            exact_match = hook.file is not None and file_name == hook.file
            suffix_match = (
                hook.file_suffix is not None
                and normalized_file.endswith(hook.file_suffix)
            )
            if exact_match or suffix_match:
                matched.append(hook)
        return matched


_RUNTIME = HotHookRuntime()


def install_if_enabled() -> bool:
    return _RUNTIME.install_if_enabled()


def reset_for_test() -> None:
    _RUNTIME.reset()


def runtime() -> HotHookRuntime:
    return _RUNTIME
