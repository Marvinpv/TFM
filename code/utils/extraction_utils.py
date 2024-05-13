import tensorflow as tf
import sqlite3
import logging
from settings import logger,LOCAL_DB_PATH
from settings import extraction_path


def create_db_cursor(db_path=LOCAL_DB_PATH):
    con = sqlite3.connect(db_path)

    return con.cursor()

def get_all_instruments(db_cursor=None):
    if not db_cursor:
        db_cursor = create_db_cursor()
    
    res = db_cursor.execute("SELECT DISTINCT instrument FROM solo_info")
    
    return [r[0] for r in res.fetchall()]


def extract_solo_info_from_melid(melid,cursor=None):
    if not cursor:
        cursor = create_db_cursor()
    
    if isinstance(melid,list):
        melid = ','.join(map(str,melid))

    res = cursor.execute(f"SELECT performer,title,instrument FROM solo_info \
                         WHERE melid IN ({melid})")
    
    row = res.fetchone()

    if len(row) == 0:
        logger.warning(f'No rows where found for melid: {melid}')
        

    return {
        'performer': row[0],
        'title': row[1],
        'instrument': row[2]  
    }

def get_instrument_solo_list(para_options):
    tfrecord_filename = extraction_path + para_options + ".tfrecord"
    display_list = []
    features = get_features_from_tfrecord(tfrecord_filename)
    db_cursor = create_db_cursor()
    for feature in features:
        melid = feature['melid'].numpy()
        solo_info = extract_solo_info_from_melid(melid)
        display_list.append((f'{str(melid).zfill(3)} - {str(solo_info['performer']).center(30)} - {solo_info['title']}',melid))
    
    return display_list