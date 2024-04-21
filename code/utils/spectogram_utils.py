import tensorflow as tf
import numpy as np
import logging



import IPython.display as display

logger = logging.getLogger(__name__)

def spectogram_to_tf_feature(spect):
    if isinstance(spect,list):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[np.array(spect)]))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=spect.numpy().flatten()))

def get_spectogram_float_list(spect):
    return tf.train.FloatList(value=spect.numpy().flatten())



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


    

def serialize_spectogram_with__melid(spect,melid,spect_shape=None):
    
    if spect_shape == None:
        spect_shape = tf.shape(spect)

    tf_feature_spect = spectogram_to_tf_feature(spect)

    features = {
        'melid': tf.train.Feature(int64_list=tf.train.Int64List(value=[melid])),
        'spectogram': tf_feature_spect,
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=spect_shape.numpy().flatten()))
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

def save_spectogram_list_to_tf_record(filepath, spect_list,melids,shapes = None):
    if not shapes:
        shapes = [tf.shape(spect) for spect in spect_list]
    
    with tf.io.TFRecordWriter(filepath) as writer:
        try:
            for spect,melid,shape in zip(spect_list,melids,shapes):
                serialized_string = serialize_spectogram_with__melid(spect,melid,shape)
                writer.write(serialized_string)
                logger.debug(f'Saved melid {melid} into {filepath}')
        except Exception as e:
            logger.error(e)


def get_features_from_tfrecord(filepath):
    raw_spectogram_dataset = tf.data.TFRecordDataset(filepath)

    spectogram_feature_description = {
        'melid': tf.io.FixedLenFeature([], tf.int64),
        'spectogram': tf.io.FixedLenFeature([], tf.float32),
        'shape': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_spectogram_function(example_proto):
        return tf.io.parse_single_example(example_proto,spectogram_feature_description)

    parsed_spectogram_dataset = raw_spectogram_dataset.map(_parse_spectogram_function)

    return parsed_spectogram_dataset




