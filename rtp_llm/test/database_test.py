from types import SimpleNamespace
from unittest import TestCase, main, mock

from rtp_llm.utils import database


class DatabaseTest(TestCase):
    def test_should_disable_fastsafetensors_shm_when_shm_is_too_small(self):
        fake_stat = SimpleNamespace(f_frsize=4096, f_bavail=8)
        with mock.patch.object(database.os, "statvfs", return_value=fake_stat):
            self.assertFalse(database._should_use_fastsafetensors_shm(64))

    def test_should_enable_fastsafetensors_shm_when_shm_is_large_enough(self):
        fake_stat = SimpleNamespace(f_frsize=4096, f_bavail=1024 * 1024)
        with mock.patch.object(database.os, "statvfs", return_value=fake_stat):
            self.assertTrue(database._should_use_fastsafetensors_shm(64))

    def test_create_fastsafetensors_loader_falls_back_without_shm(self):
        calls = []

        class FakeLoader:
            def __init__(self, **kwargs):
                calls.append(dict(kwargs))
                if kwargs.get("use_shm"):
                    raise RuntimeError("shm init failed")
                self.use_shm = kwargs.get("use_shm")

        loader = database._create_fastsafetensors_loader(
            FakeLoader,
            {"device": "cuda:0", "use_shm": True},
        )

        self.assertFalse(loader.use_shm)
        self.assertEqual(
            calls,
            [
                {"device": "cuda:0", "use_shm": True},
                {"device": "cuda:0", "use_shm": False},
            ],
        )


if __name__ == "__main__":
    main()
