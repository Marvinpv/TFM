from mt3.datasets import DatasetConfig,InferEvalSplit
from settings import extraction_path
import os
import tensorflow as tf

JAZZ_SOLOS_CONFIG = DatasetConfig(
    name='jazz_solos',
    paths={
        'train':
            'gs://tfm-jazz-transcription-marvin/solos/jazz_solos.tfrecord-?????-of-00026',
        'train_subset':
            'gs://tfm-jazz-transcription-marvin/solos/jazz_solos.tfrecord-000[01]?-of-00026',
        'validation':
            'gs://tfm-jazz-transcription-marvin/solos/jazz_solos.tfrecord-0002[123]-of-00026',
        'validation_subset':
            'gs://tfm-jazz-transcription-marvin/solos/jazz_solos.tfrecord-00021-of-00026',
        'test':
            'gs://tfm-jazz-transcription-marvin/solos/jazz_solos.tfrecord-0002[456]-of-00026'
    },
    features={
        'audio': tf.io.VarLenFeature(dtype=tf.float32),
        'sequence': tf.io.FixedLenFeature([],dtype=tf.string),
        'id': tf.io.FixedLenFeature([],dtype=tf.string),
        'sample_rate':tf.io.FixedLenFeature([],dtype=tf.int64)
    },
    train_split='train',
    train_eval_split='validation_subset',
    infer_eval_splits=[
        InferEvalSplit(name='train',suffix='eval_train_full',
                       include_in_mixture=False),
        InferEvalSplit(name='train_subset',suffix='eval_train'),
        InferEvalSplit(name='validation',suffix='validation_full',
                       include_in_mixture=False),
        InferEvalSplit(name='validation_subset',suffix='validation'),
        InferEvalSplit(name='test',suffix='test',include_in_mixture=False)
        
    ]

)