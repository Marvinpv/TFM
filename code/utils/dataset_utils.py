from settings import extraction_path
from utils.spectogram_utils import get_features_from_mt3_dir,save_samples_and_midi_to_tf_record,get_features_from_mt3_tfrecord
import os 
import tensorflow as tf
import logging
import note_seq
from mt3 import note_sequences
from scripts.dataset_solos_creation import solo_extraction_pipeline_mt3
import pandas as pd

def train_test_split(dataset_path= extraction_path):
    
    parsed_dataset = get_features_from_mt3_dir(dataset_path)
    shuffled_dataset = parsed_dataset.shuffle(buffer_size=10000)

    dataset_size = sum([1 for _ in shuffled_dataset])
    train_size = int(0.7 * dataset_size)
    validation_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - validation_size

    train_dataset = shuffled_dataset.take(train_size)
    remaining_dataset = shuffled_dataset.skip(train_size)
    val_dataset = remaining_dataset.take(validation_size)
    test_dataset = remaining_dataset.skip(validation_size)

    num_files_train = train_size//10
    if train_size % 10 != 0:
        num_files_train += 1
    num_files_test = test_size//10
    if test_size % 10 != 0:
        num_files_test += 1
    num_files_val = validation_size//10
    if validation_size % 10 != 0:
        num_files_val += 1


    batch_train = train_dataset.batch(10)
    for i,record in zip(range(num_files_train),batch_train):
        filepath = f'jazz_solos_train.tfrecord-{str((i)).zfill(5)}-of-{str(num_files_train-1).zfill(5)}'
        filepath = os.path.join(extraction_path,filepath)
        save_samples_and_midi_to_tf_record(filepath,
                                           tf.sparse.to_dense(record['audio']),
                                           record['sequence'].numpy(),
                                           record['id'].numpy(),
                                           record['sample_rate'].numpy()
                                           )
        logging.info(f'Saved train batch nr {i} into {filepath}\n')
    
    batch_val = val_dataset.batch(10)
    for i,record in zip(range(num_files_train),batch_val):
        filepath = f'jazz_solos_val.tfrecord-{str((i)).zfill(5)}-of-{str(num_files_val-1).zfill(5)}'
        filepath = os.path.join(extraction_path,filepath)
        save_samples_and_midi_to_tf_record(filepath,
                                           tf.sparse.to_dense(record['audio']),
                                           record['sequence'].numpy(),
                                           record['id'].numpy(),
                                           record['sample_rate'].numpy()
                                           )
        logging.info(f'Saved validation batch nr {i} into {filepath}\n')

    batch_test = test_dataset.batch(10)
    for i,record in zip(range(num_files_train),batch_test):
        filepath = f'jazz_solos_test.tfrecord-{str((i)).zfill(5)}-of-{str(num_files_test-1).zfill(5)}'
        filepath = os.path.join(extraction_path,filepath)
        save_samples_and_midi_to_tf_record(filepath,
                                           tf.sparse.to_dense(record['audio']),
                                           record['sequence'].numpy(),
                                           record['id'].numpy(),
                                           record['sample_rate'].numpy()
                                           )
        logging.info(f'Saved test batch nr {i} into {filepath}\n')

def restructure_dataset(dataset_path=extraction_path):
    
    parsed_dataset = get_features_from_mt3_tfrecord(os.path.join(dataset_path,'jazz_solos.tfrecord-00021-of-00026'))
    dataset_size = sum([1 for _ in parsed_dataset])
    audios = []
    seqs = []
    ids = []
    srs = []
    for i,record in zip(range(dataset_size),parsed_dataset):
        if (i%10 == 0 and i > 0) or i == dataset_size - 1:
            name_file = f'jazz_solos.tfrecord-{str((i//10-1 if i < dataset_size-1 else i//10)).zfill(5)}-of-{str(dataset_size//10).zfill(5)}'
            filepath = os.path.join(extraction_path,name_file)
            save_samples_and_midi_to_tf_record(filepath,
                                               audios,
                                               seqs,
                                               ids,
                                               srs
                                               )
            audios = []
            seqs = []
            ids = []
            srs = []
        try:
            ns = note_seq.midi_to_note_sequence(record['sequence'].numpy())
            note_sequences.validate_note_sequence(ns)
            audios.append(tf.sparse.to_dense(record['audio']))
            seqs.append(ns.SerializeToString())
            ids.append(record['id'].numpy())
            srs.append(record['sample_rate'].numpy())
        except Exception as e:
            logging.info(e)
    num_files = len(audios)//10
    if len(audios) % 10 != 0:
        num_files += 1


TEST_CSV = '/home/marvin/US/TFM/code/scripts/csv_test.csv'
TEST_DATASET_PATH= '/home/marvin/US/TFM/code/dataset/test'
def create_test_dataset():
    df_test = df = pd.read_csv(TEST_CSV)
    solo_extraction_pipeline_mt3(df_test,TEST_DATASET_PATH)
        


if __name__ == '__main__':
    '/run/media/marvin/50145B98145B7FC0/Users/Marvin/Documents/dataset3'
    restructure_dataset()
    