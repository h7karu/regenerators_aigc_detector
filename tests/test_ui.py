import unittest

from PIL import Image

from aigc_detector.ui import confidence_labels, prepare_image, verdict_markdown


class UiHelpersTest(unittest.TestCase):
    def test_confidence_labels_are_complementary(self):
        labels = confidence_labels(0.8)
        self.assertAlmostEqual(labels["AI-GENERATED"], 0.8)
        self.assertAlmostEqual(labels["REAL"], 0.2)

    def test_confidence_is_clamped(self):
        self.assertEqual(confidence_labels(2.0)["AI-GENERATED"], 1.0)
        self.assertEqual(confidence_labels(-1.0)["AI-GENERATED"], 0.0)

    def test_prepare_image_converts_mode_and_limits_size(self):
        image = Image.new("RGBA", (4096, 1024))
        prepared = prepare_image(image)
        self.assertEqual(prepared.mode, "RGB")
        self.assertEqual(prepared.size, (2048, 512))

    def test_prepare_image_rejects_missing_input(self):
        with self.assertRaisesRegex(ValueError, "upload an image"):
            prepare_image(None)

    def test_verdict_uses_larger_probability(self):
        self.assertIn("AI-GENERATED", verdict_markdown(0.75, 0.1))
        self.assertIn("REAL", verdict_markdown(0.25, 0.1))


if __name__ == "__main__":
    unittest.main()

