"""Metadata-only launcher checks; no native runtime or network is required."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from rtp_llm.flexlb.tools.mock_bundle import whale_frontend_fixture as launcher


class MetadataLauncherTest(unittest.TestCase):
    def launch(self, checkpoint, tokenizer):
        called = []
        env = {'RTP_LLM_MOCK_FRONTEND': '1', 'MOCK_BUNDLE_ROLE': 'vit',
               'CHECKPOINT_PATH': checkpoint, 'TOKENIZER_PATH': tokenizer}
        with patch.dict(os.environ, env, clear=True), \
                patch.object(launcher, 'start_mock_vit', side_effect=lambda model: called.append(str(model))), \
                patch.object(sys, 'argv', ['test']), \
                patch.object(sys, 'path', list(sys.path)):
            launcher.main()
            actual = (os.environ['CHECKPOINT_PATH'], os.environ['TOKENIZER_PATH'])
        return called[0], actual

    def test_empty_checkpoint_uses_real_tokenizer_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.json').write_text('{}')
            argv, actual = self.launch('', directory)
            self.assertEqual(actual, (directory, directory))
            self.assertEqual(argv, directory)

    def test_distinct_model_and_tokenizer_are_preserved(self):
        with tempfile.TemporaryDirectory() as model, tempfile.TemporaryDirectory() as tokenizer:
            for directory in (model, tokenizer):
                (Path(directory) / 'config.json').write_text('{}')
            _, actual = self.launch(model, tokenizer)
            self.assertEqual(actual, (model, tokenizer))

    def test_missing_real_config_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'lacks config.json'):
                self.launch('', directory)


if __name__ == '__main__':
    unittest.main()
