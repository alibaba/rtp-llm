import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import q36_nonshared_launch_contract as contract


class PortableContractTest(unittest.TestCase):
    def test_verified_dataset_and_manifest_then_reject_tampered_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data.jsonl'
            data.write_text('{"source_row_index":7,"question":"x"}\n')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'binding_scope': 'full-provider-ddp-live-source-verified',
                                            'parameter_count': 20943, 'distinct_hf_source_count': 693}))
            sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            env = {'RTP_LLM_RL_MODE': '1', 'RTP_EP_SIZE': '1',
                   'RLLM_PYTHON_TOOL_TRAIN_JSONL': str(data),
                   'Q36_DATASET_SHA256': hashlib.sha256(data.read_bytes()).hexdigest(),
                   'Q36_SR_MANIFEST': str(manifest), 'Q36_SR_MANIFEST_SHA256': sha}
            with patch.object(contract, 'SR_MANIFEST_SHA', sha), patch.object(contract, 'validate_megatron_config'):
                contract.validate({}, env)
                data.write_text('{"source_row_index":7,"question":"changed"}\n')
                with self.assertRaisesRegex(ValueError, 'dataset'):
                    contract.validate({}, env)
                with self.assertRaisesRegex(ValueError, 'independent RTP'):
                    contract.validate({}, {**env, 'RTP_RL_SHARED_BACKING': '1'})


if __name__ == '__main__':
    unittest.main()
