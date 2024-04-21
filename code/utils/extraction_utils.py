import tensorflow as tf
import sqlite3
import logging
from settings import logger,LOCAL_DB_PATH


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

