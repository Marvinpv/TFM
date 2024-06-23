import functools
from typing import Optional, Sequence

from mt3 import datasets
from mt3 import event_codec
from mt3 import metrics
from mt3 import mixing
from mt3 import preprocessors
from mt3 import run_length_encoding
from mt3 import spectrograms
from mt3 import vocabularies
from mt3.tasks import construct_task_name,add_transcription_task_to_registry
from ml.dataset_config import JAZZ_SOLOS_CONFIG

import note_seq
import numpy as np
import seqio
import t5
import tensorflow as tf


# Just use default spectrogram config.
SPECTROGRAM_CONFIG = spectrograms.SpectrogramConfig()

# Create two vocabulary configs, one default and one with only on-off velocity.
VOCAB_CONFIG_FULL = vocabularies.VocabularyConfig()
VOCAB_CONFIG_NOVELOCITY = vocabularies.VocabularyConfig(num_velocity_bins=1)


add_transcription_task_to_registry(
    dataset_config=JAZZ_SOLOS_CONFIG,
    spectrogram_config=SPECTROGRAM_CONFIG,
    vocab_config=VOCAB_CONFIG_NOVELOCITY,
    tokenize_fn=functools.partial(
        preprocessors.tokenize_transcription_example,
        audio_is_samples=True,
        id_feature_key='id'),
    onsets_only=False,
    include_ties=False)