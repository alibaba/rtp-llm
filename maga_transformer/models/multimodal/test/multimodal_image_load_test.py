from maga_transformer.models.multimodal.multimodal_common import ImageEmbeddingInterface

import os
import PIL
from unittest import TestCase, main
from PIL import Image, ImageFile
from random import randint
import tempfile
import pillow_avif
import pillow_heif

class ImageLoadTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_embedding = ImageEmbeddingInterface()

    def test(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Image.new('RGBA', (1024, 768))
            pixels = image.load()
            for i in range(1024):
                for j in range(768):
                    pixels[i, j] = (randint(0, 255), randint(0, 255), randint(0, 255))
            image.save(temp_dir + '/test.png')
            with open(temp_dir + '/test.png', "rb") as f:
                data = f.read()[:-20000]
            with open(temp_dir + '/test.png', "wb") as f:
                f.write(data)

            image.save(temp_dir + '/test.avif')
            image.save(temp_dir + '/test.heic')

            try:
                self.image_embedding._mm_preprocess(temp_dir + '/test.png')
                self.image_embedding._mm_preprocess(temp_dir + '/test.avif')
                self.image_embedding._mm_preprocess(temp_dir + '/test.heic')

                self.assertTrue(isinstance(Image.open(temp_dir + '/test.png'), PIL.PngImagePlugin.PngImageFile))
                self.assertTrue(isinstance(Image.open(temp_dir + '/test.avif'), pillow_avif.AvifImagePlugin.AvifImageFile))
                self.assertTrue(isinstance(Image.open(temp_dir + '/test.heic'), pillow_heif.as_plugin.HeifImageFile))

            except Exception as e:
                self.fail(str(e))
    
        test_trunc_image_path = '/mnt/nas1/testdata/test_trunc_image.png'
        if os.path.exists(test_trunc_image_path):
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            with self.assertRaises(PIL.UnidentifiedImageError):
                Image.open(test_trunc_image_path)
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            Image.open(test_trunc_image_path)

if __name__ == "__main__":
    main()