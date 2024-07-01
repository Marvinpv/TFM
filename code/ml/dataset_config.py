from mt3.datasets import DatasetConfig,InferEvalSplit
from settings import extraction_path
import os
import tensorflow as tf

JAZZ_SOLOS_CONFIG = DatasetConfig(
    name='jazz_solos',
    paths={
        'train':
            os.path.join(extraction_path,'jazz_solos.tfrecord-?????-of-00026'),
        'train_subset':
            os.path.join(extraction_path,'jazz_solos.tfrecord-000[01]?-of-00026'),
        'validation':
            os.path.join(extraction_path,'jazz_solos.tfrecord-0002[012]-of-00026'),
        'validation_subset':
            os.path.join(extraction_path,'jazz_solos.tfrecord-00021-of-00026'),
        'test':
            os.path.join('/home/marvin/US/TFM/code/dataset/test/','jazz_solos.tfrecord-00000-of-00000')
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