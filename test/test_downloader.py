import os
import tempfile
import unittest

import deepbee.downloader


class TestDownloader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_image = ('BEE_HOPE_GIMONDE_2016_03_23_BL2_G_FILE0690.JPG,'
                            'https://cloud.ipb.pt/d/aa29c989ab1944aaa222/files/'
                            '?p=%2FDS-COMB-PT%2FBEE_HOPE%20GIMONDE%202016_03_23%20BL2_G%20FILE0690.JPG&dl=1')

    def test_get_urls(self):
        expected = [deepbee.downloader.Image(*self.sample_image.split(','))]
        obtained = deepbee.downloader.get_urls([self.sample_image])
        self.assertEqual(expected, obtained)

    def test_download(self):
        def download_image(_, output_image):
            with open(output_image, 'w'):
                pass

        # monkey patching the urlretrieve method
        urlretrieve = deepbee.downloader.urlretrieve
        deepbee.downloader.urlretrieve = download_image

        with tempfile.TemporaryDirectory() as output_path:
            img_urls = deepbee.downloader.get_urls([self.sample_image])
            deepbee.downloader.download_images(img_urls, output_path, 0)
            self.assertIn(self.sample_image.split(',')[0], os.listdir(output_path))

        deepbee.downloader.urlretrieve = urlretrieve
