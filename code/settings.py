import logging
import IPython.display as display
import os

logger = logging.getLogger(__name__)

test_wav_filepath = "/home/marvin/US/TFM/code/tests/tests_files/Benny Carter, Oscar Peterson ft. Joe Pass - Just Friends.wav"
test_tf_record_filepath = "/home/marvin/US/TFM/code/tests/tests_files/just_friends.tfrecord"

LOCAL_DB_PATH = '/home/marvin/US/ApuntesTFM/DB/wjazzd.db' 

SAMPLING_RATE = 22000
TMP_DIR_PATH = os.path.join(os.getcwd(),'tmp')