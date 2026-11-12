import subprocess
import sys
import textwrap
import unittest


class OptionalJitDependenciesTest(unittest.TestCase):
    def test_real_polling_and_snapshot_round_trip_without_optional_packages(self):
        for missing in (['watchdog'], ['zstandard'], ['watchdog', 'zstandard']):
            with self.subTest(missing=missing):
                # Block imports before the FIRST import of the production modules.
                # The child uses the real polling worker, packer and restorer.
                code = textwrap.dedent('''
                    import importlib.abc
                    import os
                    import sys
                    import tempfile
                    import time
                    from dataclasses import replace
                    from pathlib import Path
                    from unittest import mock

                    missing = set(sys.argv[1:])
                    class MissingDependency(importlib.abc.MetaPathFinder):
                        def find_spec(self, fullname, path=None, target=None):
                            if fullname.split('.')[0] in missing:
                                raise ModuleNotFoundError(fullname, name=fullname)
                    sys.meta_path.insert(0, MissingDependency())
                    import rtp_llm.utils.jit_cache_manager as jit
                    import rtp_llm.utils.jit_cache_store as store
                    if 'watchdog' in missing:
                        assert jit.Observer is None
                    if 'zstandard' in missing:
                        assert store.zstd is None
                        assert store.SNAPSHOT_SUFFIX.endswith('.tar.gz')

                    def until(predicate):
                        deadline = time.monotonic() + 10
                        while not predicate():
                            if time.monotonic() >= deadline:
                                raise AssertionError('JIT polling did not publish')
                            time.sleep(0.01)

                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        (root / 'remote').mkdir()
                        local = root / 'scope'
                        component = replace(
                            next(c for c in jit.COMPONENTS if c.name == 'triton'),
                            local_dir=local / 'triton',
                        )
                        scope = jit.Scope('optional-deps-test', local, (component,))
                        manager = jit.JitCacheManager(scope, str(root / 'remote'))
                        with mock.patch.object(jit, 'SYNC_POLL_S', 0.02), \
                             mock.patch.object(jit, 'EARLY_WAKE_MIN_GAP_S', 0.01):
                            try:
                                assert manager.bootstrap(5)
                                assert component.local_dir.is_dir()
                                until(lambda: manager._seen is not None)
                                artifact = component.local_dir / 'kernel.cubin'
                                artifact.write_bytes(b'compiled-test-artifact')
                                expected_ns = 1700000000123456789
                                os.utime(artifact, ns=(expected_ns, expected_ns))
                                def snapshots():
                                    return list(manager.store.remote_root.glob('*' + store.SNAPSHOT_SUFFIX))
                                until(snapshots)
                                restored = manager.store.prepare_restore(root / 'staging')
                                assert restored is not None
                                result = restored.staging / 'triton/kernel.cubin'
                                assert result.read_bytes() == artifact.read_bytes()
                                assert result.stat().st_mtime_ns == expected_ns
                            finally:
                                manager.stop()
                            assert not manager._worker.is_alive()
                ''')
                subprocess.run([sys.executable, '-c', code, *missing], check=True, timeout=30)


if __name__ == '__main__':
    unittest.main()
