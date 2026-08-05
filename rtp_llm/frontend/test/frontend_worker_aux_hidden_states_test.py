import os
import pathlib
import tempfile
import unittest
from unittest import mock

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.frontend.frontend_worker import FrontendWorker


class FrontendWorkerAuxHiddenStatesTest(unittest.TestCase):
    def test_dump_returns_ready_path_with_request_metadata(self):
        worker = FrontendWorker.__new__(FrontendWorker)
        config = GenerateConfig(
            return_aux_hidden_states=True,
            aux_hidden_states_prefill_only=True,
            aux_hidden_states_layers=[1, 14, 28],
        )
        hidden_states = torch.randn(3, 12, dtype=torch.bfloat16)
        layer_ids = torch.tensor([1, 14, 28], dtype=torch.int32)
        input_ids = torch.tensor([[10, 20, 30]], dtype=torch.int32)

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(
                os.environ,
                {"AUX_HIDDEN_STATES_READY_DIR": temp_dir},
                clear=False,
            ):
                ready_path = worker._dump_aux_hidden_states_if_enabled(
                    request_id=12345,
                    generate_text="",
                    finished=True,
                    generate_config=config,
                    aux_info=None,
                    aux_hidden_states=hidden_states,
                    aux_hidden_states_layers=layer_ids,
                    input_ids=input_ids,
                    output_ids=None,
                )

            self.assertIsNotNone(ready_path)
            path = pathlib.Path(ready_path)
            self.assertTrue(path.is_absolute())
            self.assertTrue(path.name.endswith(".pt.ready"))
            self.assertIn("_req12345_", path.name)

            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["request_id"], 12345)
            self.assertTrue(payload["aux_hidden_states_prefill_only"])
            torch.testing.assert_close(payload["aux_hidden_states"], hidden_states)
            torch.testing.assert_close(payload["aux_hidden_states_layers"], layer_ids)
            torch.testing.assert_close(payload["input_ids"], input_ids)


if __name__ == "__main__":
    unittest.main()
