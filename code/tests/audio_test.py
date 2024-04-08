from unittest import TestCase,main
from utils import audio_utils as audio_r
import os
import numpy as np
import tensorflow as tf

class audioUtilsTest(TestCase):

    def test_audio_download(self):
        url = 'https://www.youtube.com/watch?v=tXe6Q5FQhQM'
        start_time = 37
        end_time = 77
        filepath = audio_r.return_in_memory_wav_audio(url,start_time,end_time)

        self.assertEqual(filepath.endswith('.wav'),True,'The downloaded file is not a wav file')
        self.assertEqual(os.path.exists(filepath),True,f'File {filepath} was not created')

    def test_create_spectogram_from_wav(self):
        filepath = "/home/marvin/US/TFM/code/tmp/Charlie Parker - Billie's Bounce.wav"
        self.assertTrue(os.path.exists(filepath),f'File {filepath} does not exist')

        spec = audio_r.create_spectogram_from_wav(filepath)

        self.assertEqual(type(spec),type(np.ndarray([])))

    def test_create_sample_with_tensorflow(self):
        filepath = "/home/marvin/US/TFM/code/tmp/Charlie Parker - Billie's Bounce.wav"

        samples,_ = audio_r.convert_audio_to_samples(filepath)

        self.assertEqual(samples.shape,(2254728,),f"Expected samples shape was (2254728,), but got {samples.shape}")

    def test_create_spectogram_from_tf_samples(self):

        filepath = "/home/marvin/US/TFM/code/tmp/Charlie Parker - Billie's Bounce.wav"

        samples,sampling_rate = audio_r.convert_audio_to_samples(filepath)
        spect,_ = audio_r.create_spectogram_from_tf_samples(samples,sampling_rate=sampling_rate)

        self.assertEqual(spect.shape, tf.TensorShape([17614,129,1]), f'Expected shape was [17614,129,1], but got {spect.shape}')
        
    def test_delete_wav_after_download(self):
        

        url = 'https://www.youtube.com/watch?v=MjI5IjsUcSs'
        start_time = 37
        end_time = 77
        filepath = audio_r.return_in_memory_wav_audio(url,start_time,end_time)

        self.assertTrue(os.path.exists(filepath),'The downloaded file does not exist')

        audio_r.delete_wav_file(filepath)

        self.assertTrue(not os.path.exists(filepath),'The downloaded file was not removed')



if __name__ == "__main__":
    main()
        

