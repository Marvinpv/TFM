from note_seq import audio_io
import os
from scipy import signal
from librosa.feature import melspectrogram
from librosa.display import specshow
import matplotlib.pyplot as plt
import yt_dlp
import io
import tensorflow as tf
import numpy as np
import logging
import settings

SAMPLING_RATE = settings.SAMPLING_RATE
TMP_DIR_PATH = settings.TMP_DIR_PATH
logger = logging.getLogger(__name__)


def create_spectogram_from_wav(in_memory_audio):
    samples = audio_io.load_audio(in_memory_audio,SAMPLING_RATE)
    spectogram = melspectrogram(y=samples,sr=SAMPLING_RATE)
    return spectogram

def return_in_memory_wav_audio(url,start_time,end_time, dir_path=TMP_DIR_PATH):
    ydl_ops = {
        'format': 'bestaudio/best',
        'download_ranges': yt_dlp.utils.download_range_func(None,[(start_time,end_time)]),
        'outtmpl': f'{TMP_DIR_PATH}/%(title)s.%(ext)s'
    }

    ydl_ops['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        }]

    with yt_dlp.YoutubeDL(ydl_ops) as ydl:
        try:
            info = ydl.extract_info(url)
        except Exception as e:
            print('Error:',e)

    return info['requested_downloads'][0]['filepath']


def save_to_TFRecord(spec, tfrec_filepath):
    return None


def convert_audio_to_samples(filepath,sr=SAMPLING_RATE):
    bin_audio = tf.io.read_file(filepath)
    audio_samples,sampling_rate = tf.audio.decode_wav(contents=bin_audio,desired_channels=1)
    #print(audio_samples.shape)
    return tf.squeeze(audio_samples,axis=-1),int(sampling_rate)

def create_spectogram_from_tf_samples(samples, sampling_rate = SAMPLING_RATE):

    waveform = tf.cast(samples, dtype=tf.float32)
    spectrogram = tf.signal.stft(
      waveform,frame_length=2048,frame_step=int(0.25*2048),fft_length=None,pad_end=True)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]

    return spectrogram,sampling_rate

# https://www.tensorflow.org/tutorials/audio/simple_audio?hl=es-419#convert_waveforms_to_spectrograms
def plot_spectrogram(spectrogram, ax = None):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  plt.pcolormesh(X, Y, log_spec)

def delete_wav_file(filepath):
    if not os.path.exists(filepath):
        logger.error(f'File {filepath} does not exist')
        return None

    try:
        os.remove(filepath)
    except Exception as e:
        logger.error(e)
    








if __name__=='__main__':
    import spectogram_utils

    filepath = "/home/marvin/US/TFM/code/tmp/Lester Young - Dickie's Dream (1939).wav"

    samples,sr = convert_audio_to_samples(filepath)
    spectogram,_ = create_spectogram_from_tf_samples(samples,sampling_rate=sr)
    mel_spect = spectogram_utils.spectogram_to_mel_scale(spectogram)
    #plot_spectrogram(spectogram)
    specshow(np.squeeze(mel_spect.numpy()).T)#,y_axis='log')
    plt.show()


