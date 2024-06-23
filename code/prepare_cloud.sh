
git clone -b feature/google-cloud-execution --single-branch https://github.com/Marvinpv/TFM.git

gsutil cp -r gs://tfm-jazz-transcription-marvin/pip_lib /home/marvin/TFM/code/pip_lib/
pip install pip_lib/mt3-0.0.1-py3-none-any.whl
pip install pip_lib/tensorflow_text-2.13.0-cp312-cp312-linux_x86_64.whl
git clone --branch=main https://github.com/google-research/t5x
cd t5x
python3 -m pip install -e '.[tpu]' -f \
  https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r code/requirements.txt

export T5X_DIR=t5x/
export PROJECT_DIR=TFM/code/



