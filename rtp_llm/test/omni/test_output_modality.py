import unittest

from rtp_llm.omni.config.output_modality import OutputModality


class TestOutputModality(unittest.TestCase):
    def test_flag_values(self):
        self.assertEqual(OutputModality.TEXT.value, 1)
        self.assertEqual(OutputModality.AUDIO.value, 2)
        self.assertEqual(OutputModality.IMAGE.value, 4)
        self.assertEqual(OutputModality.VIDEO.value, 8)

    def test_flag_combination(self):
        combined = OutputModality.TEXT | OutputModality.AUDIO
        self.assertIn(OutputModality.TEXT, combined)
        self.assertIn(OutputModality.AUDIO, combined)
        self.assertNotIn(OutputModality.IMAGE, combined)

    def test_all_modalities(self):
        all_mod = OutputModality.TEXT | OutputModality.AUDIO | OutputModality.IMAGE | OutputModality.VIDEO
        self.assertIn(OutputModality.TEXT, all_mod)
        self.assertIn(OutputModality.VIDEO, all_mod)


if __name__ == "__main__":
    unittest.main()
