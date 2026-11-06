"""Isolated math-code worker. Linux seccomp and no_new_privs are mandatory.

This is a bounded mathematical Python subset, not a general-purpose container.
No network, process creation, host credential paths, or persistent writes are
available after the policy is installed. Landlock optionally permits read-only
runtime imports; when unavailable, seccomp denies every new filesystem access.
Failure to install either mandatory policy is fail-closed.
"""

from __future__ import annotations

import ast
import contextlib
import ctypes
import ctypes.util
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import sysconfig
import time
import types


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (8, 9))
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def _landlock() -> int:
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("The math tool requires Linux Landlock")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    # Landlock uses the same syscall numbers on Linux x86_64 and aarch64.
    create, add, restrict = 444, 445, 446
    abi = libc.syscall(create, ctypes.c_void_p(), ctypes.c_size_t(0), ctypes.c_uint(1))
    if abi < 1:
        raise RuntimeError(f"Landlock unavailable: errno={ctypes.get_errno()}")
    handled = (1 << 13) - 1
    if abi >= 2:
        handled |= 1 << 13  # REFER
    if abi >= 3:
        handled |= 1 << 14  # TRUNCATE

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]

    ruleset = Ruleset(handled)
    fd = libc.syscall(create, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    if fd < 0:
        raise RuntimeError(f"Landlock create failed: errno={ctypes.get_errno()}")
    roots = {
        sys.prefix,
        sys.base_prefix,
        str(Path(sys.executable).resolve().parent.parent),
        sysconfig.get_path("stdlib"),
        sysconfig.get_path("platstdlib"),
        sysconfig.get_path("purelib"),
        sysconfig.get_path("platlib"),
        os.getcwd(),
        "/usr/lib", "/usr/lib64", "/lib", "/lib64",
        "/usr/share/zoneinfo", "/etc/ld.so.cache", "/etc/localtime",
        "/dev/null", "/dev/urandom", "/sys/devices/system/cpu",
        "/proc/cpuinfo", "/proc/meminfo", "/proc/self/stat",
        "/proc/self/status", "/proc/self/maps",
    }
    try:
        seen = set()
        for root in roots:
            if not root:
                continue
            p = Path(root).resolve()
            if not p.exists() or str(p) in seen:
                continue
            seen.add(str(p))
            path_fd = os.open(p, os.O_PATH | os.O_CLOEXEC)
            try:
                access = (1 << 2) | ((1 << 3) if p.is_dir() else 0)
                rule = PathRule(access, path_fd)
                if libc.syscall(add, fd, 1, ctypes.byref(rule), 0) != 0:
                    raise RuntimeError(f"Landlock path rule failed: errno={ctypes.get_errno()}")
            finally:
                os.close(path_fd)
        if libc.syscall(restrict, fd, 0) != 0:
            raise RuntimeError(f"Landlock restrict failed: errno={ctypes.get_errno()}")
    finally:
        os.close(fd)
    return int(abi)


def _no_new_privs() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise RuntimeError(f"Cannot set no_new_privs: errno={ctypes.get_errno()}")


def _seccomp(lib, *, deny_files: bool) -> None:
    # The library is loaded before Landlock restricts filesystem access.
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x7FFF0000)  # ALLOW by default; filesystem is Landlocked.
    if not ctx:
        raise RuntimeError("seccomp_init failed")
    denied = (
        "socket", "socketpair", "connect", "accept", "accept4", "bind", "listen",
        "sendto", "sendmsg", "sendmmsg", "recvfrom", "recvmsg", "recvmmsg",
        "setsockopt", "getsockopt", "shutdown", "execve", "execveat", "fork",
        "vfork", "clone", "clone3", "ptrace", "process_vm_readv", "process_vm_writev",
        "mount", "umount2", "unshare", "setns", "keyctl", "add_key", "request_key",
        "open_by_handle_at", "bpf", "userfaultfd", "perf_event_open", "io_uring_setup",
        "io_uring_enter", "io_uring_register", "reboot", "kexec_load", "init_module",
        "finit_module", "delete_module",
        "chmod", "fchmod", "fchmodat", "fchmodat2", "chown", "fchown", "lchown",
        "fchownat", "utime", "utimes", "futimesat", "utimensat",
        "kill", "tkill", "tgkill", "rt_sigqueueinfo", "rt_tgsigqueueinfo",
        "pidfd_open", "pidfd_send_signal", "pidfd_getfd", "kcmp", "process_madvise",
        "shmget", "shmat", "shmdt", "shmctl", "msgget", "msgsnd", "msgrcv", "msgctl",
        "semget", "semop", "semtimedop", "semctl", "ipc", "mq_open", "mq_unlink",
        "mq_timedsend", "mq_timedreceive", "mq_notify", "mq_getsetattr",
        "memfd_create", "memfd_secret", "setpgid", "setsid", "setrlimit", "prlimit64",
        "sched_setaffinity", "sched_setscheduler", "sched_setparam", "setpriority",
        "setuid", "setgid", "setreuid", "setregid", "setresuid", "setresgid",
        "setfsuid", "setfsgid", "setgroups", "capset", "syslog", "socketcall",
    )
    if deny_files:
        denied += (
            "open", "openat", "openat2", "creat", "name_to_handle_at",
            "stat", "lstat", "newfstatat", "statx", "access", "faccessat", "faccessat2",
            "readlink", "readlinkat", "getdents", "getdents64", "chdir", "fchdir",
            "unlink", "unlinkat", "rename", "renameat", "renameat2", "link", "linkat",
            "symlink", "symlinkat", "mkdir", "mkdirat", "rmdir", "mknod", "mknodat",
            "truncate", "ftruncate", "fallocate", "ioctl",
        )
    try:
        for name in denied:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and lib.seccomp_rule_add(ctx, 0x00050001, number, 0) != 0:
                raise RuntimeError(f"Cannot deny syscall {name}")
        if lib.seccomp_load(ctx) != 0:
            raise RuntimeError("seccomp_load failed")
    finally:
        lib.seccomp_release(ctx)


class _Imports(ast.NodeTransformer):
    def visit_Import(self, node):
        result = []
        for alias in node.names:
            if alias.name not in ("sympy", "math"):
                raise ValueError("Only import sympy and import math are supported")
            target = alias.asname or alias.name
            if target.startswith("_"):
                raise ValueError("Private names are not supported")
            result.append(ast.Assign(targets=[ast.Name(id=target, ctx=ast.Store())], value=ast.Name(id=alias.name, ctx=ast.Load())))
        return result

    def visit_ImportFrom(self, node):
        if node.module not in ("sympy", "math") or node.level:
            raise ValueError("Only math and sympy imports are supported")
        result = []
        for alias in node.names:
            if alias.name == "*" or alias.name.startswith("_"):
                raise ValueError("Wildcard and private imports are not supported")
            result.append(ast.Assign(
                targets=[ast.Name(id=alias.asname or alias.name, ctx=ast.Store())],
                value=ast.Attribute(value=ast.Name(id=node.module, ctx=ast.Load()), attr=alias.name, ctx=ast.Load()),
            ))
        return result


def _safe_tree(code: str):
    if len(code) > 12000:
        raise ValueError("Code exceeds the 12000-character limit")
    tree = ast.parse(code, mode="exec")
    forbidden = (ast.ClassDef, ast.With, ast.AsyncWith, ast.AsyncFunctionDef, ast.Await, ast.Global, ast.Nonlocal, ast.Delete)
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("Private names are not supported")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private attributes are not supported")
    tree = _Imports().visit(tree)
    return ast.fix_missing_locations(tree)


class _BoundedOutput(io.StringIO):
    def write(self, text):
        if self.tell() + len(text) > 16000:
            raise RuntimeError("Tool output exceeded 16000 characters")
        return super().write(text)


def _preload_sympy():
    """Warm trusted common paths before a no-files policy blocks lazy imports."""
    import sympy as sp

    x, y = sp.symbols("x y")
    warmed_results = [
        sp.solve(sp.Eq(x**2, 2), x), sp.solve("x**2 - 2", x),
        sp.factor(x**4 - 1), sp.integrate(sp.sin(x), x),
        sp.integrate(1 / (x**2 + 1), x),
        sp.integrate(sp.exp(-x**2), (x, 0, sp.oo)), sp.diff(sp.sin(x), x),
        sp.simplify(sp.sin(x)**2 + sp.cos(x)**2),
        sp.linsolve([x + y - 3, x - y - 1], (x, y)),
        sp.nonlinsolve([x**2 - 1], [x]), sp.summation(x, (x, 1, 10)),
        sp.limit(sp.sin(x) / x, x, 0), sp.Matrix([[1, 2], [3, 4]]).eigenvals(),
        sp.factorint(360), sp.nroots(x**3 - 2), sp.S("1/2"),
    ]
    # Warm the printers too: printing an otherwise preloaded result must not
    # need to open a Python module after the strict no-files policy is active.
    for value in warmed_results:
        str(value)
        repr(value)
    return sp


def main() -> None:
    sys.dont_write_bytecode = True
    started = time.monotonic()
    result = {"ok": False, "stdout": "", "error": "", "security": {}}
    try:
        request = json.loads(sys.stdin.read(20000))
        tree = _safe_tree(str(request.get("code", "")))
        _limits()
        sympy = _preload_sympy()
        library_name = ctypes.util.find_library("seccomp") or "libseccomp.so.2"
        lib = ctypes.CDLL(library_name, use_errno=True)
        _no_new_privs()
        abi, landlock_error = 0, None
        try:
            abi = _landlock()
        except Exception as exc:
            landlock_error = str(exc)
        # This fallback is stricter than Landlock: all new opens, path metadata
        # reads, and filesystem mutations are denied. Only preloaded libraries
        # and the three inherited stdio pipes are available to the program.
        _seccomp(lib, deny_files=not bool(abi))
        result["security"] = {
            "mode": "landlock_seccomp" if abi else "seccomp_no_files",
            "landlock_abi": abi, "landlock_error": landlock_error,
            "no_new_privs": True, "seccomp": True, "network": "denied",
            "filesystem": "read-only allowlist" if abi else "all new opens and path access denied",
            "signals_and_external_ipc": "denied",
        }
        try:
            probe = socket.socket()
            probe.close()
            raise RuntimeError("Network policy self-test unexpectedly allowed a socket")
        except OSError as exc:
            if exc.errno != 1:
                raise RuntimeError(f"Network policy self-test returned unexpected errno {exc.errno}")
        try:
            with open("/etc/passwd", "rb") as probe:
                probe.read(1)
            raise RuntimeError("Filesystem policy self-test unexpectedly allowed /etc/passwd")
        except PermissionError:
            pass
        result["security"]["network_probe_denied"] = True
        result["security"]["host_file_probe_denied"] = True
        # No open/eval/exec/__import__/getattr/globals/locals are exposed to the program.
        builtin_names = (
            "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
            "frozenset", "int", "isinstance", "len", "list", "map", "max", "min",
            "next", "pow", "print", "range", "repr", "reversed", "round", "set",
            "slice", "sorted", "str", "sum", "tuple", "zip", "Exception", "ValueError",
        )
        import builtins

        safe_builtins = {name: getattr(builtins, name) for name in builtin_names}
        sympy_names = (
            "Symbol", "symbols", "Integer", "Rational", "Float", "Eq", "Ne", "Lt",
            "Le", "Gt", "Ge", "solve", "solveset", "linsolve", "nonlinsolve", "simplify",
            "factor", "expand", "cancel", "together", "apart", "gcd", "lcm", "isprime",
            "factorint", "divisors", "totient", "binomial", "factorial", "fibonacci",
            "sqrt", "root", "real_root", "sin", "cos", "tan", "asin", "acos", "atan",
            "log", "exp", "diff", "integrate", "summation", "product", "limit",
            "Matrix", "eye", "zeros", "ones", "Poly", "degree", "nroots", "N",
            "S", "pi", "E", "I", "oo", "Interval", "FiniteSet", "Union", "Intersection",
            "And", "Or", "Not", "Mod", "ceiling", "floor", "Min", "Max", "Abs",
        )
        safe_sympy = types.SimpleNamespace(**{name: getattr(sympy, name) for name in sympy_names})
        namespace = {"__builtins__": safe_builtins, "sympy": safe_sympy, "sp": safe_sympy, "math": math}
        output = _BoundedOutput()
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(tree, "<python_math_tool>", "exec"), namespace, namespace)
            result["ok"] = True
        finally:
            result["stdout"] = output.getvalue()
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    result["elapsed_s"] = round(time.monotonic() - started, 4)
    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
