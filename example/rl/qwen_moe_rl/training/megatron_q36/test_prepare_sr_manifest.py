import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_sr_manifest as schema


class SchemaMaterializationTest(unittest.TestCase):
    def test_qualified_schema_and_corruption_rejection(self):
        encoded = schema.SOURCE.read_bytes()
        raw = schema.decode_manifest(encoded)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), schema.MANIFEST_SHA)
        with self.assertRaises((ValueError, TypeError)):
            schema.decode_manifest(b'not-the-qualified-archive')
        with self.assertRaisesRegex(ValueError, 'encoded length'):
            schema.decode_manifest(encoded + b'x')

    def test_exclusive_output_and_cleanup_of_new_partial_file(self):
        encoded = schema.SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'schema.json'
            schema.materialize(encoded, output)
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                schema.materialize(encoded, output)
            self.assertEqual(output.read_bytes(), before)
            partial = Path(directory) / 'partial.json'
            with patch.object(schema.os, 'fsync', side_effect=OSError('injected fsync failure')):
                with self.assertRaisesRegex(OSError, 'injected fsync failure'):
                    schema.materialize(encoded, partial)
            self.assertFalse(partial.exists())


if __name__ == '__main__':
    unittest.main()
