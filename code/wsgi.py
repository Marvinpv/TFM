from flask import Flask,render_template,jsonify,request,Blueprint
from settings import spect_records_path
from utils.spectogram_utils import create_spectogram_for_melid,get_features_from_tfrecord
from utils.extraction_utils import create_db_cursor,extract_solo_info_from_melid
from librosa.display import specshow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os


## Preprocess solo list for dataset
#ins_dic = {}
#for ins in os.listdir('./dataset'):
#    ins_dic[ins[:ins.find('.')]]


app = Flask('APP',static_url_path='/static')

app.jinja_env.filters['zip'] = zip

@app.route("/")
def main_menu():
    return render_template('main_menu.html')

@app.route("/dataset")
def dataset_view():
    return render_template('dataset_view.html')

@app.context_processor
def obtener_opciones():
    def get_instrument_solo_list(para_options):
        tfrecord_filename = spect_records_path + para_options + ".tfrecord"
        display_list = []
        features = get_features_from_tfrecord(tfrecord_filename)
        db_cursor = create_db_cursor()
        for feature in features:
            melid = feature['melid'].numpy()
            solo_info = extract_solo_info_from_melid(melid)
            display_list.append((f'{str(melid).zfill(3)} - {str(solo_info['performer']).center(30)} - {solo_info['title']}',melid))

        return display_list

    return dict(get_instrument_solo_list=get_instrument_solo_list)




@app.route('/mostrar_info', methods=['POST'])
def mostrar_info():
    instrument = request.form['ins']
    melid = int(request.form['melid'])
    performer = request.form['performer']
    song_name = request.form['song_name']
    if os.path.exists('./static/images/spect_tmp.png'):
        os.remove('./static/images/spect_tmp.png')
    create_spectogram_for_melid(melid,instrument)
    return render_template('spectogram_popup.html',performer=performer,song_name=song_name)


@app.route("/getimage")
def get_img():
    return "monito.jpg"
