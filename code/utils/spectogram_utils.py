import tensorflow as tf
import numpy as np
import logging
from settings import extraction_path,spect_records_path
from utils.extraction_utils import create_db_cursor
from librosa.display import specshow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
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

def serialize_samples_and_midi_with__melid(sample,midi,melid,sr):

    if(isinstance(sample,list)):
        print('Hola')

    features = {
        'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(melid)])),
        'audio': tf.train.Feature(float_list=tf.train.FloatList(value=sample.numpy())),
        'sequence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[midi])),
        'sample_rate': tf.train.Feature(int64_list=tf.train.Int64List(value=[sr]))
    }

    spect_example = tf.train.Example(features=tf.train.Features(feature=features))
    return spect_example.SerializeToString()

def save_samples_and_midi_to_tf_record(filepath,samples_list,midi_list,melid_list,sr):

    with tf.io.TFRecordWriter(filepath) as writer:
        try:
            for sample,midi,melid,curr_sr in zip(samples_list,midi_list,melid_list,sr):
                serialized_string = serialize_samples_and_midi_with__melid(sample,midi,melid,curr_sr)
                writer.write(serialized_string)
                logger.debug(f'Saved melid {melid} into {filepath}')
        except Exception as e:
            logger.error(e)

def get_features_from_mt3_tfrecord(filepath):
    raw_input_dataset = tf.data.TFRecordDataset(filepath)
    input_feature_description = {
        'id': tf.io.FixedLenFeature([],tf.string),
        'audio': tf.io.VarLenFeature(tf.float32),
        'sequence': tf.io.FixedLenFeature([],tf.string),
        'sample_rate': tf.io.FixedLenFeature([],tf.int64)
    }

    def _parse_input_function(example_proto):
        return tf.io.parse_single_example(example_proto,input_feature_description)

    parsed_input_dataset = raw_input_dataset.map(_parse_input_function)

    return parsed_input_dataset

def get_features_from_mt3_dir(dirpath = extraction_path):
    files = [os.path.join(dirpath,arch) for arch in os.listdir(dirpath)]
    raw_input_dataset = tf.data.TFRecordDataset(files)
    input_feature_description = {
        'id': tf.io.FixedLenFeature([],tf.string),
        'audio': tf.io.VarLenFeature(tf.float32),
        'sequence': tf.io.FixedLenFeature([],tf.string),
        'sample_rate': tf.io.FixedLenFeature([],tf.int64)
    }

    def _parse_input_function(example_proto):
        return tf.io.parse_single_example(example_proto,input_feature_description)

    parsed_input_dataset = raw_input_dataset.map(_parse_input_function)

    return parsed_input_dataset


def get_features_from_tfrecord(filepath):
    raw_spectogram_dataset = tf.data.TFRecordDataset(filepath)
    spectogram_feature_description = {
        'melid': tf.io.FixedLenFeature([], tf.int64),
        'spectogram': tf.io.VarLenFeature(tf.float32),
        'shape': tf.io.FixedLenFeature([2,], tf.int64),
    }

    def _parse_spectogram_function(example_proto):
        return tf.io.parse_single_example(example_proto,spectogram_feature_description)

    parsed_spectogram_dataset = raw_spectogram_dataset.map(_parse_spectogram_function)

    return parsed_spectogram_dataset

def create_spectogram_for_melid(melid,para_options):
        tfrecord_filename = spect_records_path + para_options + ".tfrecord"
        display_list = []
        features = get_features_from_tfrecord(tfrecord_filename)
        db_cursor = create_db_cursor()
        for feature in features:
            current_melid = feature['melid'].numpy()
            if current_melid == melid:
                mel_spect = tf.sparse.to_dense(feature['spectogram']).numpy().reshape(feature['shape'].numpy())
                specshow(np.squeeze(mel_spect).T)#,y_axis='log')
                plt.savefig('static/images/spect_tmp.png')
                plt.close()


