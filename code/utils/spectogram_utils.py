import tensorflow as tf
import numpy as np
import logging



import IPython.display as display

logger = logging.getLogger(__name__)

def spectogram_to_tf_feature(spect : tf.Tensor):
    return tf.train.Feature(float_list=tf.train.FloatList(value=spect.numpy().flatten()))


def spectogram_to_mel_scale(spectogram,
                            num_mel_bins=64,
                            sample_rate=48000):
    
    if len(spectogram.shape) == 3:
        spectogram = tf.squeeze(spectogram, axis=-1)

    num_spect_bins = int(spectogram.shape[-1])

    mel_mat = tf.signal.linear_to_mel_weight_matrix(num_mel_bins,\
                                                    num_spect_bins,
                                                    sample_rate)

    mel_spect = tf.tensordot(spectogram,mel_mat,1)
    mel_spect.set_shape(spectogram.shape[:-1].concatenate(mel_mat.shape[-1:]))

    return tf.math.log(mel_spect)


    

def serialize_spectogram_with__melid(spect,melid):
    
    spect_shape = tf.shape(spect)
    tf_feature_spect = spectogram_to_tf_feature(spect)

    features = {
        'melid': tf.train.Feature(int64_list=tf.train.Int64List(value=[melid])),
        'spectogram': tf_feature_spect,
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=spect_shape))
    }

    spect_example = tf.train.Example(features=tf.train.Features(feature=features))
    return spect_example.SerializeToString()



def save_to_tf_record(tf_record,filepath):
    with tf.io.TFRecordWriter(filepath) as writer:
        try:
            writer.write(tf_record)
            logger.debug(f'Saved TFRecord in {filepath}')
        except Exception as e:
            logger.error(e)

