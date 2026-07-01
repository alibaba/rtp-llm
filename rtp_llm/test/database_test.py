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

    def test_load_tensors_by_prefix_falls_back_when_shm_loader_init_fails(self):
        class FakeCkptFile:
            file_name = "model-00001.safetensors"

            def get_tensor_names(self):
                return ["decoder.weight"]

            def load_tensors(self, device, direct_io):
                self.last_load_args = (device, direct_io)
                return {"decoder.weight": "direct-load"}

        class FailingLoadWithShm:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("shm init failed")

        db = database.CkptDatabase(None)
        db.pretrain_file_list = [FakeCkptFile()]
        db.finetune_file_list = []

        with mock.patch.object(
            database, "_should_use_fastsafetensors_shm", return_value=True
        ), mock.patch.dict(
            "sys.modules",
            {"fast_safetensors": SimpleNamespace(LoadWithShm=FailingLoadWithShm)},
        ):
            result = db.load_tensors_by_prefix(("decoder",), "cuda:0", True)

        self.assertEqual(result, {"decoder.weight": ["direct-load"]})
        self.assertEqual(
            db.pretrain_file_list[0].last_load_args,
            ("cuda:0", True),
        )

    def test_load_tensors_by_prefix_falls_back_when_shm_loader_load_fails(self):
        class FakeCkptFile:
            file_name = "model-00001.safetensors"

            def get_tensor_names(self):
                return ["decoder.weight"]

            def load_tensors(self, device, direct_io):
                self.last_load_args = (device, direct_io)
                return {"decoder.weight": "direct-load"}

        class FailingLoadWithShm:
            def __init__(self, *args, **kwargs):
                pass

            def load_safetensors_to_device(self, file_name):
                raise RuntimeError(f"shm load failed for {file_name}")

        db = database.CkptDatabase(None)
        db.pretrain_file_list = [FakeCkptFile()]
        db.finetune_file_list = []

        with mock.patch.object(
            database, "_should_use_fastsafetensors_shm", return_value=True
        ), mock.patch.dict(
            "sys.modules",
            {"fast_safetensors": SimpleNamespace(LoadWithShm=FailingLoadWithShm)},
        ):
            result = db.load_tensors_by_prefix(("decoder",), "cuda:0", False)

        self.assertEqual(result, {"decoder.weight": ["direct-load"]})
        self.assertEqual(
            db.pretrain_file_list[0].last_load_args,
            ("cuda:0", False),
        )


if __name__ == "__main__":
    main()
