import importlib
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from rtp_llm.utils import aiter_jit_patch


def _publish_module_under_aiter_lock(
    lock_path, so_path, published_mtime_ns, acquired_event, release_event
):
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        acquired_event.set()
        if not release_event.wait(10):
            raise RuntimeError("timed out waiting to publish test AITER module")
        with open(so_path, "w") as file:
            file.write("fresh")
        os.utime(so_path, ns=(published_mtime_ns, published_mtime_ns))
    finally:
        os.close(lock_fd)
        os.remove(lock_path)


def _hold_owned_aiter_lock(so_path, acquired_event, release_event):
    with aiter_jit_patch._aiter_build_lock(so_path):
        acquired_event.set()
        if not release_event.wait(10):
            raise RuntimeError("timed out waiting to release test AITER lock")


def _abandon_owned_aiter_lock(so_path, acquired_event):
    with aiter_jit_patch._aiter_build_lock(so_path):
        acquired_event.set()
        # Simulate SIGKILL/segfault semantics: process fds are closed, but the
        # context manager cannot unlink the file-baton path.
        os._exit(0)


class _FakeFileBaton:
    def __init__(self, lock_file_path, wait_seconds=0.2):
        self.lock_file_path = lock_file_path
        self.wait_seconds = wait_seconds
        self.fd = None

    def try_acquire(self):
        try:
            self.fd = os.open(
                self.lock_file_path,
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o644,
            )
            return True
        except FileExistsError:
            return False

    def wait(self):
        while os.path.exists(self.lock_file_path):
            time.sleep(self.wait_seconds)

    def release(self):
        if self.fd is not None:
            os.close(self.fd)
        os.remove(self.lock_file_path)


class AiterJitPatchTest(unittest.TestCase):
    def tearDown(self):
        aiter_jit_patch.uninstall_aiter_jit_patch()

    def test_matches_only_codegen_scripts_inside_aiter_packages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            aiter_root = os.path.join(tmpdir, "site-packages", "aiter_meta")
            os.makedirs(aiter_root)
            aiter_script = os.path.join(aiter_root, "generate.py")
            unrelated_script = os.path.join(tmpdir, "generate.py")
            with mock.patch.object(
                aiter_jit_patch,
                "_aiter_package_roots",
                return_value=(aiter_root,),
            ):
                self.assertTrue(
                    aiter_jit_patch._is_aiter_codegen_command(
                        f"{sys.executable} {aiter_script} --output_dir /tmp/out"
                    )
                )
                self.assertFalse(
                    aiter_jit_patch._is_aiter_codegen_command(
                        f"{sys.executable} {unrelated_script} --output_dir /tmp/out"
                    )
                )

    def test_rejects_compound_shell_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "generate.py")
            with mock.patch.object(
                aiter_jit_patch, "_aiter_package_roots", return_value=(tmpdir,)
            ):
                self.assertFalse(
                    aiter_jit_patch._is_aiter_codegen_command(
                        f"{sys.executable} {script} && rm -rf /tmp/unrelated"
                    )
                )

    def test_matches_bazel_runfiles_script_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            aiter_root = os.path.join(tmpdir, "runfiles", "aiter_meta")
            real_script = os.path.join(tmpdir, "external", "codegen.py")
            logical_script = os.path.join(aiter_root, "hsa", "codegen.py")
            os.makedirs(os.path.dirname(real_script))
            os.makedirs(os.path.dirname(logical_script))
            with open(real_script, "w"):
                pass
            os.symlink(real_script, logical_script)
            with mock.patch.object(
                aiter_jit_patch,
                "_aiter_package_roots",
                return_value=(aiter_root,),
            ):
                self.assertTrue(
                    aiter_jit_patch._is_aiter_codegen_command(
                        f"{sys.executable} {logical_script} --output_dir /tmp/out"
                    )
                )

    def test_codegen_command_does_not_expand_filter_globs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "generate.py")
            args_path = os.path.join(tmpdir, "args.txt")
            with open(script_path, "w") as file:
                file.write(
                    "import pathlib, sys\n"
                    f"pathlib.Path({args_path!r}).write_text('\\n'.join(sys.argv[1:]))\n"
                )

            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                open("fmha_fwd_d256_bf16_kernel.o", "w").close()
                command = (
                    f"{sys.executable} {script_path} "
                    "--filter *bf16*_nlogits* --output_dir /tmp/out"
                )
                with mock.patch.object(
                    aiter_jit_patch,
                    "_aiter_package_roots",
                    return_value=(tmpdir,),
                ):
                    ret = aiter_jit_patch._run_aiter_codegen_command(command)
            finally:
                os.chdir(cwd)

            self.assertEqual(ret, 0)
            with open(args_path) as file:
                args = file.read().splitlines()
            self.assertIn("*bf16*_nlogits*", args)
            self.assertNotIn("fmha_fwd_d256_bf16_kernel.o", args)

    def test_codegen_command_preserves_exit_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "generate.py")
            with open(script_path, "w") as file:
                file.write("import sys\nsys.exit(7)\n")

            with mock.patch.object(
                aiter_jit_patch, "_aiter_package_roots", return_value=(tmpdir,)
            ):
                ret = aiter_jit_patch._run_aiter_codegen_command(
                    f"{sys.executable} {script_path}"
                )

            self.assertEqual(ret, 7)

    def test_non_aiter_system_call_preserves_original_contract(self):
        original_system = mock.Mock(return_value=1792)
        patched_system = aiter_jit_patch._make_patched_system(original_system)

        self.assertEqual(patched_system("exit 7"), 1792)
        original_system.assert_called_once_with("exit 7")

    def test_remove_stale_aiter_jit_module(self):
        error_details = (
            "undefined symbol: fmha_batch_prefill",
            "file too short",
            "invalid ELF header",
        )
        for error_detail in error_details:
            with self.subTest(error_detail=error_detail):
                with tempfile.TemporaryDirectory() as tmpdir:
                    jit_dir = os.path.join(tmpdir, "aiter", "jit")
                    build_dir = os.path.join(jit_dir, "build", "mha_batch_prefill_bad")
                    os.makedirs(build_dir)
                    so_path = os.path.join(jit_dir, "mha_batch_prefill_bad.so")
                    with open(so_path, "w"):
                        pass

                    with mock.patch.object(
                        aiter_jit_patch,
                        "_aiter_jit_roots",
                        return_value=(jit_dir,),
                    ):
                        removed = aiter_jit_patch._maybe_remove_stale_aiter_jit_module(
                            "aiter.jit.mha_batch_prefill_bad",
                            ImportError(f"{so_path}: {error_detail}"),
                        )

                    self.assertIs(
                        removed,
                        aiter_jit_patch._StaleModuleRecovery.REBUILD,
                    )
                    self.assertFalse(os.path.exists(so_path))
                    self.assertFalse(os.path.exists(build_dir))

    def test_stale_cleanup_rejects_path_or_module_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jit_dir = os.path.join(tmpdir, "aiter", "jit")
            outside_dir = os.path.join(tmpdir, "outside")
            os.makedirs(jit_dir)
            os.makedirs(outside_dir)
            outside_so = os.path.join(outside_dir, "bad.so")
            mismatched_so = os.path.join(jit_dir, "other.so")
            for path in (outside_so, mismatched_so):
                with open(path, "w"):
                    pass

            with mock.patch.object(
                aiter_jit_patch, "_aiter_jit_roots", return_value=(jit_dir,)
            ):
                self.assertIs(
                    aiter_jit_patch._maybe_remove_stale_aiter_jit_module(
                        "aiter.jit.bad",
                        ImportError(f"{outside_so}: undefined symbol: bad"),
                    ),
                    aiter_jit_patch._StaleModuleRecovery.NOT_HANDLED,
                )
                self.assertIs(
                    aiter_jit_patch._maybe_remove_stale_aiter_jit_module(
                        "aiter.jit.bad",
                        ImportError(f"{mismatched_so}: undefined symbol: bad"),
                    ),
                    aiter_jit_patch._StaleModuleRecovery.NOT_HANDLED,
                )

            self.assertTrue(os.path.exists(outside_so))
            self.assertTrue(os.path.exists(mismatched_so))

    def test_stale_cleanup_rejects_unrelated_import_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jit_dir = os.path.join(tmpdir, "aiter", "jit")
            os.makedirs(jit_dir)
            so_path = os.path.join(jit_dir, "bad.so")
            with open(so_path, "w"):
                pass

            with mock.patch.object(
                aiter_jit_patch, "_aiter_jit_roots", return_value=(jit_dir,)
            ):
                recovery = aiter_jit_patch._maybe_remove_stale_aiter_jit_module(
                    "aiter.jit.bad",
                    ImportError(
                        f"{so_path}: libmissing.so: cannot open shared object file"
                    ),
                )

            self.assertIs(
                recovery,
                aiter_jit_patch._StaleModuleRecovery.NOT_HANDLED,
            )
            self.assertTrue(os.path.exists(so_path))

    def test_concurrent_stale_cleanup_allows_every_caller_to_rebuild(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jit_dir = os.path.join(tmpdir, "aiter", "jit")
            os.makedirs(jit_dir)
            so_path = os.path.join(jit_dir, "bad.so")
            with open(so_path, "w"):
                pass
            error = ImportError(f"{so_path}: undefined symbol: bad")

            with mock.patch.object(
                aiter_jit_patch, "_aiter_jit_roots", return_value=(jit_dir,)
            ):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(
                        executor.map(
                            lambda _: (
                                aiter_jit_patch._maybe_remove_stale_aiter_jit_module(
                                    "aiter.jit.bad", error
                                )
                            ),
                            range(8),
                        )
                    )

            self.assertEqual(
                results,
                [aiter_jit_patch._StaleModuleRecovery.REBUILD] * 8,
            )
            self.assertFalse(os.path.exists(so_path))

    def test_peer_cleanup_does_not_delete_an_in_progress_build(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jit_dir = os.path.join(tmpdir, "aiter", "jit")
            build_dir = os.path.join(jit_dir, "build", "bad")
            os.makedirs(build_dir)
            sentinel = os.path.join(build_dir, "compiling.o")
            with open(sentinel, "w"):
                pass
            so_path = os.path.join(jit_dir, "bad.so")

            with mock.patch.object(
                aiter_jit_patch, "_aiter_jit_roots", return_value=(jit_dir,)
            ):
                removed_by_peer = aiter_jit_patch._maybe_remove_stale_aiter_jit_module(
                    "aiter.jit.bad",
                    ImportError(f"{so_path}: undefined symbol: bad"),
                )

            self.assertIs(
                removed_by_peer,
                aiter_jit_patch._StaleModuleRecovery.REBUILD,
            )
            self.assertTrue(os.path.exists(sentinel))

    def test_cleanup_waits_for_aiter_builder_and_retries_fresh_module(self):
        if "fork" not in multiprocessing.get_all_start_methods():
            self.skipTest("AITER JIT is supported only on Linux/fork platforms")

        with tempfile.TemporaryDirectory() as tmpdir:
            jit_dir = os.path.join(tmpdir, "aiter", "jit")
            build_dir = os.path.join(jit_dir, "build")
            os.makedirs(build_dir)
            so_path = os.path.join(jit_dir, "bad.so")
            with open(so_path, "w") as file:
                file.write("stale")

            import_started_ns = time.time_ns()
            lock_path = os.path.join(build_dir, "lock_bad")
            context = multiprocessing.get_context("fork")
            acquired_event = context.Event()
            release_event = context.Event()
            process = context.Process(
                target=_publish_module_under_aiter_lock,
                args=(
                    lock_path,
                    so_path,
                    import_started_ns + 1_000_000,
                    acquired_event,
                    release_event,
                ),
            )
            process.start()
            try:
                self.assertTrue(acquired_event.wait(5))
                with mock.patch.object(
                    aiter_jit_patch,
                    "_aiter_jit_roots",
                    return_value=(jit_dir,),
                ):
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        recovery = executor.submit(
                            aiter_jit_patch._maybe_remove_stale_aiter_jit_module,
                            "aiter.jit.bad",
                            ImportError(f"{so_path}: undefined symbol: bad"),
                            import_started_ns,
                        )
                        time.sleep(0.1)
                        self.assertFalse(recovery.done())
                        release_event.set()
                        self.assertIs(
                            recovery.result(timeout=5),
                            aiter_jit_patch._StaleModuleRecovery.RETRY_IMPORT,
                        )
            finally:
                release_event.set()
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join(5)

            self.assertEqual(process.exitcode, 0)
            with open(so_path) as file:
                self.assertEqual(file.read(), "fresh")

    def test_reclaims_old_build_lock_from_dead_local_owner(self):
        if "fork" not in multiprocessing.get_all_start_methods():
            self.skipTest("AITER JIT is supported only on Linux/fork platforms")

        with tempfile.TemporaryDirectory() as tmpdir:
            so_path = os.path.join(tmpdir, "bad.so")
            lock_path = os.path.join(tmpdir, "build", "lock_bad")
            context = multiprocessing.get_context("fork")
            acquired_event = context.Event()
            process = context.Process(
                target=_abandon_owned_aiter_lock,
                args=(so_path, acquired_event),
            )
            process.start()
            try:
                self.assertTrue(acquired_event.wait(5))
                process.join(5)
                if process.is_alive():
                    self.fail("orphan-lock owner did not exit")
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(5)

            self.assertEqual(process.exitcode, 0)
            self.assertTrue(os.path.exists(lock_path))
            old_timestamp = time.time() - 60
            os.utime(lock_path, (old_timestamp, old_timestamp))

            started = time.monotonic()
            with aiter_jit_patch._aiter_build_lock(
                so_path,
                timeout_seconds=1,
                stale_lock_seconds=0,
                poll_seconds=0.01,
            ):
                self.assertTrue(os.path.exists(lock_path))
            self.assertLess(time.monotonic() - started, 1)
            self.assertFalse(os.path.exists(lock_path))

    def test_old_build_lock_from_live_owner_times_out_without_removal(self):
        if "fork" not in multiprocessing.get_all_start_methods():
            self.skipTest("AITER JIT is supported only on Linux/fork platforms")

        with tempfile.TemporaryDirectory() as tmpdir:
            so_path = os.path.join(tmpdir, "bad.so")
            lock_path = os.path.join(tmpdir, "build", "lock_bad")
            context = multiprocessing.get_context("fork")
            acquired_event = context.Event()
            release_event = context.Event()
            process = context.Process(
                target=_hold_owned_aiter_lock,
                args=(so_path, acquired_event, release_event),
            )
            process.start()
            try:
                self.assertTrue(acquired_event.wait(5))
                old_timestamp = time.time() - 60
                os.utime(lock_path, (old_timestamp, old_timestamp))
                original_inode = os.stat(lock_path).st_ino

                started = time.monotonic()
                with self.assertRaisesRegex(
                    aiter_jit_patch._AiterBuildLockTimeout,
                    "owner pid=.* is alive",
                ):
                    with aiter_jit_patch._aiter_build_lock(
                        so_path,
                        timeout_seconds=0.15,
                        stale_lock_seconds=0,
                        poll_seconds=0.01,
                    ):
                        self.fail("contender unexpectedly acquired a live peer lock")
                self.assertLess(time.monotonic() - started, 1)
                self.assertEqual(os.stat(lock_path).st_ino, original_inode)
            finally:
                release_event.set()
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join(5)

            self.assertEqual(process.exitcode, 0)
            self.assertFalse(os.path.exists(lock_path))

    def test_legacy_orphan_lock_fails_fast_and_propagates_from_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jit_dir = os.path.join(tmpdir, "aiter", "jit")
            build_root = os.path.join(jit_dir, "build")
            os.makedirs(build_root)
            so_path = os.path.join(jit_dir, "bad.so")
            lock_path = os.path.join(build_root, "lock_bad")
            with open(so_path, "w"):
                pass
            with open(lock_path, "w"):
                pass
            old_timestamp = time.time() - 60
            os.utime(lock_path, (old_timestamp, old_timestamp))
            original_inode = os.stat(lock_path).st_ino

            original_import = mock.Mock(
                side_effect=ImportError(f"{so_path}: undefined symbol: bad")
            )
            patched_import = aiter_jit_patch._make_patched_import_module(
                original_import
            )
            started = time.monotonic()
            with mock.patch.object(
                aiter_jit_patch, "_aiter_jit_roots", return_value=(jit_dir,)
            ), mock.patch.multiple(
                aiter_jit_patch,
                _AITER_BUILD_LOCK_TIMEOUT_SECONDS=0.15,
                _AITER_BUILD_LOCK_STALE_SECONDS=0,
                _AITER_BUILD_LOCK_POLL_SECONDS=0.01,
            ):
                with self.assertRaisesRegex(
                    aiter_jit_patch._AiterBuildLockTimeout,
                    "owner is unknown.*preserved",
                ):
                    patched_import("aiter.jit.bad")

            self.assertLess(time.monotonic() - started, 1)
            self.assertEqual(original_import.call_count, 1)
            self.assertEqual(os.stat(lock_path).st_ino, original_inode)
            self.assertTrue(os.path.exists(so_path))

    def test_lock_release_preserves_replacement_owner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            so_path = os.path.join(tmpdir, "bad.so")
            lock_path = os.path.join(tmpdir, "build", "lock_bad")
            with aiter_jit_patch._aiter_build_lock(so_path):
                owned_inode = os.stat(lock_path).st_ino
                os.remove(lock_path)
                with open(lock_path, "w") as file:
                    file.write("peer")
                self.assertNotEqual(os.stat(lock_path).st_ino, owned_inode)

            self.assertTrue(os.path.exists(lock_path))
            with open(lock_path) as file:
                self.assertEqual(file.read(), "peer")

    def test_repeated_lock_owner_changes_still_respect_deadline(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            aiter_jit_patch, "_try_create_owned_build_lock", return_value=None
        ), mock.patch.object(
            aiter_jit_patch,
            "_try_reclaim_stale_build_lock",
            return_value="lock owner changed while checking staleness",
        ):
            started = time.monotonic()
            with self.assertRaisesRegex(
                aiter_jit_patch._AiterBuildLockTimeout,
                "lock owner changed.*preserved",
            ):
                with aiter_jit_patch._aiter_build_lock(
                    os.path.join(tmpdir, "bad.so"),
                    timeout_seconds=0.05,
                    stale_lock_seconds=0,
                    poll_seconds=0.01,
                ):
                    self.fail("lock churn unexpectedly bypassed the deadline")
            self.assertLess(time.monotonic() - started, 1)

    def test_import_wrapper_retries_after_peer_rebuild(self):
        module = types.ModuleType("aiter.jit.bad")
        original_import = mock.Mock(
            side_effect=[
                ImportError("/tmp/bad.so: undefined symbol: bad"),
                module,
            ]
        )
        patched_import = aiter_jit_patch._make_patched_import_module(original_import)
        with mock.patch.object(
            aiter_jit_patch,
            "_maybe_remove_stale_aiter_jit_module",
            return_value=aiter_jit_patch._StaleModuleRecovery.RETRY_IMPORT,
        ):
            self.assertIs(patched_import("aiter.jit.bad"), module)
        self.assertEqual(original_import.call_count, 2)

    def test_temporary_hooks_restore_stdlib_on_success_and_exception(self):
        original_system = os.system
        original_import_module = importlib.import_module

        with aiter_jit_patch._temporary_stdlib_hooks():
            self.assertIsNot(os.system, original_system)
            self.assertIsNot(importlib.import_module, original_import_module)
        self.assertIs(os.system, original_system)
        self.assertIs(importlib.import_module, original_import_module)

    def test_temporary_hooks_do_not_overwrite_newer_stdlib_bindings(self):
        original_system = os.system
        original_import_module = importlib.import_module
        newer_system = mock.Mock(return_value=0)
        newer_import_module = mock.Mock()
        try:
            with aiter_jit_patch._temporary_stdlib_hooks():
                os.system = newer_system
                importlib.import_module = newer_import_module
            self.assertIs(os.system, newer_system)
            self.assertIs(importlib.import_module, newer_import_module)
        finally:
            os.system = original_system
            importlib.import_module = original_import_module

    def test_install_restores_stdlib_when_aiter_bootstrap_fails(self):
        original_system = os.system
        original_import_module = importlib.import_module

        def fail_aiter_import(name, package=None):
            if name == "aiter":
                raise RuntimeError("aiter bootstrap failed")
            return original_import_module(name, package)

        with mock.patch.object(
            importlib, "import_module", side_effect=fail_aiter_import
        ) as bootstrap_import:
            with self.assertRaisesRegex(RuntimeError, "aiter bootstrap failed"):
                aiter_jit_patch.install_aiter_jit_patch(enabled=True)
            self.assertIs(os.system, original_system)
            self.assertIs(importlib.import_module, bootstrap_import)

        with self.assertRaisesRegex(RuntimeError, "bootstrap failed"):
            with aiter_jit_patch._temporary_stdlib_hooks():
                raise RuntimeError("bootstrap failed")
        self.assertIs(os.system, original_system)
        self.assertIs(importlib.import_module, original_import_module)

    def test_install_is_local_idempotent_and_reversible(self):
        core = types.ModuleType("aiter.jit.core")
        core.os = os
        core.importlib = importlib
        core.FileBaton = _FakeFileBaton
        original_system = os.system
        original_import_module = importlib.import_module

        with mock.patch.object(
            aiter_jit_patch, "_bootstrap_aiter_core", return_value=core
        ) as bootstrap:
            self.assertTrue(aiter_jit_patch.install_aiter_jit_patch(enabled=True))
            installed_os = core.os
            installed_importlib = core.importlib
            installed_file_baton = core.FileBaton
            self.assertTrue(aiter_jit_patch.install_aiter_jit_patch(enabled=True))

        bootstrap.assert_called_once_with()
        self.assertIs(core.os, installed_os)
        self.assertIs(core.importlib, installed_importlib)
        self.assertIs(core.FileBaton, installed_file_baton)
        self.assertIsNot(core.FileBaton, _FakeFileBaton)
        self.assertIs(os.system, original_system)
        self.assertIs(importlib.import_module, original_import_module)

        self.assertTrue(aiter_jit_patch.uninstall_aiter_jit_patch())
        self.assertIs(core.os, os)
        self.assertIs(core.importlib, importlib)
        self.assertIs(core.FileBaton, _FakeFileBaton)
        self.assertFalse(aiter_jit_patch.uninstall_aiter_jit_patch())

    def test_bootstrap_file_baton_times_out_before_first_eager_jit(self):
        core = types.ModuleType("aiter.jit.core")
        core.os = os
        core.importlib = importlib
        core.FileBaton = _FakeFileBaton
        jit_compile_globals = {"FileBaton": _FakeFileBaton}
        core._jit_compile = types.FunctionType(
            (lambda: None).__code__, jit_compile_globals, "_jit_compile"
        )
        aiter_module = types.ModuleType("aiter")
        aiter_module.core = core
        qualified_file_baton = types.ModuleType("aiter.jit.utils.file_baton")
        qualified_file_baton.FileBaton = _FakeFileBaton

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "build", "lock_module_aiter_core")
            os.makedirs(os.path.dirname(lock_path))
            with open(lock_path, "w"):
                pass
            original_inode = os.stat(lock_path).st_ino

            def fake_import(name, package=None):
                if name == "aiter":
                    self.assertEqual(
                        os.environ[aiter_jit_patch._AITER_AOT_IMPORT_ENV], "1"
                    )
                    sys.modules["aiter"] = aiter_module
                    return aiter_module
                if name == "aiter.jit.utils.file_baton":
                    return qualified_file_baton
                self.fail(f"unexpected bootstrap import: {name}")

            def fake_reload(module):
                self.assertIs(module, aiter_module)
                self.assertEqual(os.environ[aiter_jit_patch._AITER_AOT_IMPORT_ENV], "0")
                self.assertIsNot(core.FileBaton, _FakeFileBaton)
                self.assertIs(core.FileBaton, jit_compile_globals["FileBaton"])
                self.assertIs(core.FileBaton, qualified_file_baton.FileBaton)
                baton = core.FileBaton(lock_path, wait_seconds=0.01)
                self.assertFalse(baton.try_acquire())
                baton.wait()
                self.fail("orphaned bootstrap baton unexpectedly became available")

            with mock.patch.dict(sys.modules, {}, clear=False):
                sys.modules.pop("aiter", None)
                sys.modules["aiter.jit.utils.file_baton"] = qualified_file_baton
                with mock.patch.dict(
                    os.environ,
                    {aiter_jit_patch._AITER_AOT_IMPORT_ENV: "0"},
                    clear=False,
                ), mock.patch.object(
                    importlib, "import_module", side_effect=fake_import
                ), mock.patch.object(
                    importlib, "reload", side_effect=fake_reload
                ) as reload_aiter, mock.patch.multiple(
                    aiter_jit_patch,
                    _AITER_BUILD_LOCK_TIMEOUT_SECONDS=0.05,
                    _AITER_BUILD_LOCK_STALE_SECONDS=0,
                ):
                    started = time.monotonic()
                    with self.assertRaisesRegex(
                        aiter_jit_patch._AiterBuildLockTimeout,
                        "owner is unknown.*preserved",
                    ):
                        aiter_jit_patch._bootstrap_aiter_core()
                    self.assertEqual(
                        os.environ[aiter_jit_patch._AITER_AOT_IMPORT_ENV], "0"
                    )
                reload_aiter.assert_called_once_with(aiter_module)

            self.assertLess(time.monotonic() - started, 1)
            self.assertEqual(os.stat(lock_path).st_ino, original_inode)
            self.assertIs(core.FileBaton, _FakeFileBaton)
            self.assertIs(jit_compile_globals["FileBaton"], _FakeFileBaton)
            self.assertIs(qualified_file_baton.FileBaton, _FakeFileBaton)

    def test_install_adopts_bootstrap_file_batons_and_restores_them(self):
        core = types.ModuleType("aiter.jit.core")
        core.os = os
        core.importlib = importlib
        core.FileBaton = _FakeFileBaton
        jit_compile_globals = {"FileBaton": _FakeFileBaton}
        core._jit_compile = types.FunctionType(
            (lambda: None).__code__, jit_compile_globals, "_jit_compile"
        )
        aiter_module = types.ModuleType("aiter")
        aiter_module.core = core
        qualified_file_baton = types.ModuleType("aiter.jit.utils.file_baton")
        qualified_file_baton.FileBaton = _FakeFileBaton

        def fake_import(name, package=None):
            if name == "aiter":
                self.assertEqual(os.environ[aiter_jit_patch._AITER_AOT_IMPORT_ENV], "1")
                sys.modules["aiter"] = aiter_module
                return aiter_module
            if name == "aiter.jit.utils.file_baton":
                return qualified_file_baton
            self.fail(f"unexpected bootstrap import: {name}")

        def fake_reload(module):
            self.assertIs(module, aiter_module)
            self.assertEqual(os.environ[aiter_jit_patch._AITER_AOT_IMPORT_ENV], "0")
            self.assertIs(core.FileBaton, jit_compile_globals["FileBaton"])
            self.assertIs(core.FileBaton, qualified_file_baton.FileBaton)
            return module

        with mock.patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("aiter", None)
            sys.modules["aiter.jit.utils.file_baton"] = qualified_file_baton
            with mock.patch.dict(
                os.environ,
                {aiter_jit_patch._AITER_AOT_IMPORT_ENV: "0"},
                clear=False,
            ), mock.patch.object(
                importlib, "import_module", side_effect=fake_import
            ), mock.patch.object(
                importlib, "reload", side_effect=fake_reload
            ) as reload_aiter:
                self.assertTrue(aiter_jit_patch.install_aiter_jit_patch(enabled=True))
                installed_file_baton = core.FileBaton
                self.assertIsNot(installed_file_baton, _FakeFileBaton)
                self.assertIs(
                    installed_file_baton,
                    jit_compile_globals["FileBaton"],
                )
                self.assertIs(
                    installed_file_baton,
                    qualified_file_baton.FileBaton,
                )
                self.assertEqual(os.environ[aiter_jit_patch._AITER_AOT_IMPORT_ENV], "0")
                self.assertTrue(aiter_jit_patch.uninstall_aiter_jit_patch())

            reload_aiter.assert_called_once_with(aiter_module)
            self.assertIs(core.FileBaton, _FakeFileBaton)
            self.assertIs(jit_compile_globals["FileBaton"], _FakeFileBaton)
            self.assertIs(qualified_file_baton.FileBaton, _FakeFileBaton)

    def test_installed_file_baton_times_out_on_legacy_orphan_without_module(self):
        core = types.ModuleType("aiter.jit.core")
        core.os = os
        core.importlib = importlib
        core.FileBaton = _FakeFileBaton

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "build", "lock_bad")
            os.makedirs(os.path.dirname(lock_path))
            with open(lock_path, "w"):
                pass
            old_timestamp = time.time() - 60
            os.utime(lock_path, (old_timestamp, old_timestamp))
            original_inode = os.stat(lock_path).st_ino
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "bad.so")))

            with mock.patch.object(
                aiter_jit_patch, "_bootstrap_aiter_core", return_value=core
            ), mock.patch.multiple(
                aiter_jit_patch,
                _AITER_BUILD_LOCK_TIMEOUT_SECONDS=0.05,
                _AITER_BUILD_LOCK_STALE_SECONDS=0,
            ):
                self.assertTrue(aiter_jit_patch.install_aiter_jit_patch(enabled=True))
                baton = core.FileBaton(lock_path, wait_seconds=0.01)
                self.assertFalse(baton.try_acquire())
                started = time.monotonic()
                with self.assertRaisesRegex(
                    aiter_jit_patch._AiterBuildLockTimeout,
                    "owner is unknown.*preserved",
                ):
                    baton.wait()

            self.assertLess(time.monotonic() - started, 1)
            self.assertEqual(os.stat(lock_path).st_ino, original_inode)

    def test_installed_file_baton_records_owner_and_releases(self):
        core = types.ModuleType("aiter.jit.core")
        core.os = os
        core.importlib = importlib
        core.FileBaton = _FakeFileBaton

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            aiter_jit_patch, "_bootstrap_aiter_core", return_value=core
        ):
            self.assertTrue(aiter_jit_patch.install_aiter_jit_patch(enabled=True))
            lock_path = os.path.join(tmpdir, "lock")
            baton = core.FileBaton(lock_path)
            self.assertTrue(baton.try_acquire())
            with open(lock_path, "rb") as file:
                owner = aiter_jit_patch._parse_build_lock_owner(file.read())
            self.assertIsNotNone(owner)
            self.assertEqual(owner.pid, os.getpid())
            baton.release()
            self.assertFalse(os.path.exists(lock_path))

    def test_install_patches_jit_compile_file_baton_binding(self):
        core = types.ModuleType("aiter.jit.core")
        core.os = os
        core.importlib = importlib
        core.FileBaton = _FakeFileBaton
        jit_compile_globals = {"FileBaton": _FakeFileBaton}
        core._jit_compile = types.FunctionType(
            (lambda: None).__code__, jit_compile_globals, "_jit_compile"
        )
        qualified_file_baton = types.ModuleType("aiter.jit.utils.file_baton")
        qualified_file_baton.FileBaton = _FakeFileBaton

        with mock.patch.dict(
            sys.modules,
            {"aiter.jit.utils.file_baton": qualified_file_baton},
        ), mock.patch.object(
            aiter_jit_patch, "_bootstrap_aiter_core", return_value=core
        ):
            self.assertTrue(aiter_jit_patch.install_aiter_jit_patch(enabled=True))
            self.assertIs(core.FileBaton, jit_compile_globals["FileBaton"])
            self.assertIs(core.FileBaton, qualified_file_baton.FileBaton)
            self.assertIsNot(core.FileBaton, _FakeFileBaton)
            self.assertTrue(aiter_jit_patch.uninstall_aiter_jit_patch())
            self.assertIs(core.FileBaton, _FakeFileBaton)
            self.assertIs(jit_compile_globals["FileBaton"], _FakeFileBaton)
            self.assertIs(qualified_file_baton.FileBaton, _FakeFileBaton)

    def test_install_can_be_disabled(self):
        with mock.patch.object(aiter_jit_patch, "_bootstrap_aiter_core") as bootstrap:
            self.assertFalse(aiter_jit_patch.install_aiter_jit_patch(enabled=False))
            with mock.patch.dict(
                os.environ,
                {"RTP_LLM_DISABLE_AITER_JIT_PATCH": "1"},
                clear=False,
            ):
                self.assertFalse(aiter_jit_patch.install_aiter_jit_patch())
        bootstrap.assert_not_called()

    def test_disabling_an_installed_patch_restores_aiter_core(self):
        core = types.ModuleType("aiter.jit.core")
        core.os = os
        core.importlib = importlib
        core.FileBaton = _FakeFileBaton
        with mock.patch.object(
            aiter_jit_patch, "_bootstrap_aiter_core", return_value=core
        ):
            self.assertTrue(aiter_jit_patch.install_aiter_jit_patch(enabled=True))

        self.assertIsNot(core.os, os)
        self.assertFalse(aiter_jit_patch.install_aiter_jit_patch(enabled=False))
        self.assertIs(core.os, os)
        self.assertIs(core.importlib, importlib)
        self.assertIs(core.FileBaton, _FakeFileBaton)

    def test_concurrent_install_bootstraps_once(self):
        core = types.ModuleType("aiter.jit.core")
        core.os = os
        core.importlib = importlib
        core.FileBaton = _FakeFileBaton
        with mock.patch.object(
            aiter_jit_patch, "_bootstrap_aiter_core", return_value=core
        ) as bootstrap:
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(
                    executor.map(
                        lambda _: aiter_jit_patch.install_aiter_jit_patch(enabled=True),
                        range(32),
                    )
                )

        self.assertEqual(results, [True] * 32)
        bootstrap.assert_called_once_with()

    def test_uninstall_does_not_overwrite_a_newer_core_binding(self):
        core = types.ModuleType("aiter.jit.core")
        core.os = os
        core.importlib = importlib
        core.FileBaton = _FakeFileBaton
        with mock.patch.object(
            aiter_jit_patch, "_bootstrap_aiter_core", return_value=core
        ):
            aiter_jit_patch.install_aiter_jit_patch(enabled=True)

        newer_os_binding = object()
        newer_file_baton = type("NewerFileBaton", (), {})
        core.os = newer_os_binding
        core.FileBaton = newer_file_baton
        self.assertTrue(aiter_jit_patch.uninstall_aiter_jit_patch())
        self.assertIs(core.os, newer_os_binding)
        self.assertIs(core.importlib, importlib)
        self.assertIs(core.FileBaton, newer_file_baton)

    def test_import_rtp_llm_does_not_change_stdlib_bindings(self):
        # Run in a fresh interpreter: this test must record identities before
        # the package's first import or the old permanent hook would look clean.
        script = f"""
import sys
sys.path[:0] = {sys.path!r}
import importlib
import os
original_system = os.system
original_import_module = importlib.import_module
import rtp_llm
assert os.system is original_system
assert importlib.import_module is original_import_module
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
