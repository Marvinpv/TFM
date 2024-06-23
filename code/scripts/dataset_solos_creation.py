import tensorflow as tf
import numpy as np
import logging
import settings
import pandas as pd
from utils import extraction_utils,spectogram_utils,audio_utils
import os
import mido

def create_dataframe_from_csv(csv_path=settings.csv_youtube_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        settings.logger.error(e)

    return df

def solo_extraction_pipeline(df=None):
    if not df:
        df = create_dataframe_from_csv()

    db_cursor = extraction_utils.create_db_cursor()
    instruments_dict = {ins:[]for ins in extraction_utils.get_all_instruments(db_cursor)}
    melids = df['melid'].unique().tolist()
    for melid in melids:
        df_id = df.loc[df['melid'] == melid]
        yt_ids = df_id['youtube_id'].to_list()
        start_times = df_id['solo_start_sec'].to_list()
        end_times = df_id['solo_end_sec'].to_list() 
        mel_info = extraction_utils.extract_solo_info_from_melid(melid,db_cursor)
        downloaded_filepath = None
        idx = 0
        while not downloaded_filepath and idx < len(yt_ids):
            url = f'https://www.youtube.com/watch?v={yt_ids[idx]}'
            downloaded_filepath = audio_utils.return_in_memory_wav_audio(url=url,start_time=start_times[idx],end_time=end_times[idx])
            idx+=1
        if downloaded_filepath:
            samples,_ = audio_utils.convert_audio_to_samples(downloaded_filepath)
            spect,_ = audio_utils.create_spectogram_from_tf_samples(samples)
            log_mel_spect = spectogram_utils.spectogram_to_mel_scale(spect)
            serialized_spect = spectogram_utils.get_spectogram_float_list(log_mel_spect)
            instruments_dict[mel_info['instrument']].append((log_mel_spect,melid,tf.shape(log_mel_spect)))
            audio_utils.delete_wav_file(downloaded_filepath)

    for k,solos in instruments_dict.items():
        if len(solos):
            spects = [solo[0] for solo in solos]
            melids = [solo[1] for solo in solos]
            shapes = [solo[2] for solo in solos]
            spectogram_utils.save_spectogram_list_to_tf_record(os.path.join(settings.spect_records_path,k)+'.tfrecord',spects,melids,shapes)


def solo_extraction_pipeline_mt3(df=None):
    if not df:
        df = create_dataframe_from_csv()

    db_cursor = extraction_utils.create_db_cursor()
    melids = df['melid'].unique().tolist()
    audios = []
    midi = []
    instruments = []
    final_melids = []
    srs = []
    for melid in melids[:10]:
        df_id = df.loc[df['melid'] == melid]
        yt_ids = df_id['youtube_id'].to_list()
        start_times = df_id['solo_start_sec'].to_list()
        end_times = df_id['solo_end_sec'].to_list()
          
        mel_info = extraction_utils.extract_solo_info_from_melid(melid,db_cursor)
        downloaded_filepath = None
        idx = 0
        while not downloaded_filepath and idx < len(yt_ids):
            url = f'https://www.youtube.com/watch?v={yt_ids[idx]}'
            downloaded_filepath = audio_utils.return_in_memory_wav_audio(url=url,start_time=start_times[idx],end_time=end_times[idx])
            idx+=1
        if downloaded_filepath:
            audio_samples,sr = audio_utils.convert_audio_to_samples(downloaded_filepath)
            audio_utils.delete_wav_file(downloaded_filepath)
            midi_file_name = df_id['db'].iloc[0][:df_id['db'].iloc[0].upper().index('_SOLO_Q')] + '_FINAL.mid'
            midi_path = os.path.join(settings.midi_filepath,midi_file_name)
            if os.path.exists(midi_path) and audio_samples != None:
                with open(os.path.join(settings.midi_filepath,midi_file_name),'rb') as midi_file:
                    try:
                        midi_bytestring = midi_file.read()
                        audios.append(audio_samples)
                        midi.append(midi_bytestring)
                        instruments.append(mel_info['instrument'])
                        final_melids.append(melid)
                        srs.append(sr)
                    except Exception as e:
                        logging.exception(e)
            else:
                melids.remove(melid)
        else:
            melids.remove(melid)
            
    if len(midi) != len(final_melids):
        print('cabesa')   

    num_tfrecords = len(midi)//10
    if len(midi)%10 != 0:
        num_tfrecords+=1
    for i in range(num_tfrecords):
        filepath = f'jazz_solos.tfrecord-{str(i).zfill(5)}-of-{str(num_tfrecords-1).zfill(5)}'
        filepath = os.path.join(settings.extraction_path,filepath)
        spectogram_utils.save_samples_and_midi_to_tf_record(filepath,audios[i*10:i*10+10],midi[i*10:i*10+10],final_melids[i*10:i*10+10],srs[i*10:i*10+10])

if __name__ == '__main__':
    solo_extraction_pipeline_mt3()