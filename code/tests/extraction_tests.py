from unittest import TestCase,main
from utils import audio_utils,spectogram_utils,extraction_utils
import os
import numpy as np
import tensorflow as tf


class ExtractionUtilsTest(TestCase):

    def test_extract_info_from_melid(self):
        test_melid = 3
        query = extraction_utils.extract_solo_info_from_melid(test_melid)

        self.assertEqual(query['performer'],['Art Pepper'])
        self.assertEqual(query['title'],['Desafinado'])
        self.assertEqual(query['instrument'],['as'])

    