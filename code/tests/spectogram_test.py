from unittest import TestCase,main
from utils import audio_utils,spectogram_utils
import os
import numpy as np
import tensorflow as tf
import settings

test_wav_filepath = settings.test_wav_filepath
test_tf_record_filepath = settings.test_tf_record_filepath

class SpectogramUtilsTest(TestCase):

    def test_spectogram_to_tf_feature(self):
        samples,_ = audio_utils.convert_audio_to_samples(test_wav_filepath)
        spect,_ = audio_utils.create_spectogram_from_tf_samples(samples)

        feat = spectogram_utils.spectogram_to_tf_feature(spect)

        self.assertEqual(type(feat),tf.train.Feature)
        self.assertGreater(len(feat.float_list.value),0)
    
    def test_serialize_tf_feature_to_string(self):
        samples,_ = audio_utils.convert_audio_to_samples(test_wav_filepath)
        spect,_ = audio_utils.create_spectogram_from_tf_samples(samples)

        feature_string = spectogram_utils.serialize_spectogram_with__melid(spect,10)

        self.assertGreater(len(feature_string),0, "Serialized feature is empty")
        self.assertEqual(type(feature_string),bytes,"Serialized object is not a byte-string")
    
    def test_save_to_tf_record(self):
        samples,_ = audio_utils.convert_audio_to_samples(test_wav_filepath)
        spect,_ = audio_utils.create_spectogram_from_tf_samples(samples)

        feature_string = spectogram_utils.serialize_spectogram_with__melid(spect,10)

        spectogram_utils.save_to_tf_record(feature_string,test_tf_record_filepath)

        self.assertTrue(os.path.exists(test_tf_record_filepath),"Created tfRecord does not exist")
    
    


if __name__ == "__main__":
    main()