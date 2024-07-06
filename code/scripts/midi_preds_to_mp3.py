import os
from utils.audio_utils import midi_to_wav, wav_to_mp3


if __name__ == '__main__':
    # Rutas de los archivos

    midi_paths = [os.path.join('static/audios/',mid_file) for mid_file in os.listdir('static/audios/') if mid_file.endswith('.mid')]
    wav_paths = [wav_file[:wav_file.index('.mid')]+'.wav' for wav_file in midi_paths]
    mp3_paths = [mp3_file[:mp3_file.index('.mid')]+'.mp3' for mp3_file in midi_paths]


    # Convertir MIDI a MP3
    for midi_path,wav_path in zip(midi_paths,wav_paths):
        midi_to_wav(midi_path, wav_path) 

    for wav_path,mp3_path in zip(wav_paths,mp3_paths):
        wav_to_mp3(wav_path, mp3_path) 