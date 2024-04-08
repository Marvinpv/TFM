import tensorflow as tf
import sqlite3
import logging
from settings import logger,LOCAL_DB_PATH


def create_db_cursor(db_path=LOCAL_DB_PATH):
    con = sqlite3.connect(db_path)

    return con.cursor()

def extract_solo_info_from_melid(melid,cursor=None):
    if not cursor:
        cursor = create_db_cursor()
    
    if isinstance(melid,list):
        melid = ','.join(map(str,melid))

    res = cursor.execute(f"SELECT performer,title,instrument FROM solo_info \
                         WHERE melid IN ({melid})")
    
    rows = res.fetchall()

    if len(rows) == 0:
        logger.warning(f'No rows where found for melid: {melid}')
        

    return {
        'performer':[row[0] for row in rows],
        'title':[row[1] for row in rows],
        'instrument':[row[2] for row in rows] 
    }

