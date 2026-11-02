"""Optional smoke evidence reads host metadata only and is inert when disabled."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


path=Path(__file__).parents[1]/'models_py/utils/smoke_evidence.py'
spec=importlib.util.spec_from_file_location('k31_smoke_evidence',path)
evidence=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


class SmokeEvidenceTest(unittest.TestCase):
    def test_disabled_path_does_not_touch_params(self):
        with patch.object(evidence,'_ENABLED',False):
            evidence.completed_dsa_chunks('indexer',None,2,128)

    def test_warmup_and_decode_are_not_prefill_evidence(self):
        with patch.object(evidence,'_ENABLED',True),patch.object(evidence.logging,'getLogger') as logger:
            evidence.completed_dsa_chunks('sfa',SimpleNamespace(is_warmup=True),2,128)
            evidence.completed_dsa_chunks('sfa',SimpleNamespace(is_warmup=False,is_prefill=False),2,128)
            logger.assert_not_called()

    def test_completed_metadata_has_actual_chunk_count(self):
        import json
        params=SimpleNamespace(is_warmup=False,is_prefill=True,shard_rank=3,shard_size=8,
            query_cp=True,indexer_execution=SimpleNamespace(fp8_indexer=True))
        with patch.object(evidence,'_ENABLED',True),patch.object(evidence.logging,'getLogger') as logger:
            evidence.completed_dsa_chunks('indexer',params,3,8192)
            value=json.loads(logger.return_value.info.call_args.args[1])
        self.assertEqual(value['completed_chunks'],3)
        self.assertEqual(value['local_query_rows'],8192)
        self.assertEqual(value['shard_rank'],3)
        self.assertTrue(value['fp8_indexer'])


if __name__=='__main__':
    unittest.main()
