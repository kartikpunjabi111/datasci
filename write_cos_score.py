import io
import logging
import socket
import sys
import time
from datetime import datetime
from logging.handlers import SysLogHandler

import numpy
import pandas as pd
import requests
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

import ML

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
mapping = pd.read_csv('/home/ubuntu/recsys/cos_cpu/mapping_state_city.csv')

PAPERTRAIL_HOST = "Credentials"
PAPERTRAIL_PORT = Credentials


class ContextFilter(logging.Filter):
    hostname = socket.gethostname()

    def filter(self, record):
        record.hostname = ContextFilter.hostname
        return True


def init_logger(mod_value):
    syslog = SysLogHandler(address=(PAPERTRAIL_HOST, PAPERTRAIL_PORT))
    syslog.addFilter(ContextFilter())

    format = '%(asctime)s MLRecoWriter RecoWriter_' + str(mod_value) + ': %(message)s'
    formatter = logging.Formatter(format, datefmt='%b %d %H:%M:%S')
    syslog.setFormatter(formatter)

    logger = logging.getLogger()
    logger.addHandler(syslog)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)


def adapt_array(array):
    """
    Using the numpy.save function to save a binary version of the array,
    and BytesIO to catch the stream of data and convert it into a BLOB.
    """

    out = io.BytesIO()
    numpy.save(out, array)
    out.seek(0)

    return out.read()


def convert_array(blob):
    """
    Using BytesIO to convert the binary version of the array back into a numpy array.
    """
    if blob == b'0':
        return 0
    out = io.BytesIO(blob)
    out.seek(0)

    return numpy.load(out)


def calc_sim(vec1, vec2):
    if type(vec1) == int or type(vec2) == int:
        return 0
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def calc_sim2(vec1, vec2):
    if type(vec1) == int or type(vec2) == int:
        return 0
    if type(vec2)==bytes:
        return 0
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def preprocess(content, mod_value):
    return [( content[i][0], convert_array(content[i][1]), convert_array(content[i][2]),content[i][3],content[i][4] )
            for i in range(len(content)) if content[i][0] % 4 == mod_value] 


# Rename
def pre(user):
    new_none = []
    new_not_none = []
    for i in tqdm(range(len(user))):
        if user[i][2] == None:
            new_none.append((user[i][0], convert_array(user[i][1])))
        else:
            new_not_none.append((user[i][0], convert_array(user[i][1]), convert_array(user[i][2]) , user[i][3] , user[i][4] , user[i][5] , user[i][6] ) )
    return new_none, new_not_none

#filtering for locations in city-town-state hierarchy
def get_locations(location):
    location=location.decode()   
    a=mapping[mapping.state==str(location)].city.unique()
    if len(a)>0:
        return a
    a=mapping[mapping.city==str(location)].town.unique()
    if len(a)>0:
        if a[0]==int or type(a[0])==str:
            return a
    return location

#decreasing the pool wrt filtered locations
def get_pool( content_data , user_data , content_clustered ):
    locations=[]
    user_specific_content_id=[]

    for i in range(3,7):
        locations.append(get_locations(user_data[i]))
      
    #converted-location
    convert_loc=[]
    
    for loc in locations:
        if type(loc)==numpy.ndarray:
            convert_loc=convert_loc+list(loc)
        else:   convert_loc.append(loc)
    list(set(convert_loc))

    for loc in convert_loc:
        if type(loc)==float:
            break
        loc=loc.encode()
        
        if loc in content_clustered.keys():
            user_specific_content_id.append(content_clustered[loc])
            
    user_specific_content_id=[a[0] for a in user_specific_content_id]
    user_specific_content_id=list(set(user_specific_content_id))

    return [( content_data[i][0],content_data[i][1],content_data[i][2], content_data[i][3],content_data[i][4] ) for i in range(len(content_data)) if  content_data[i][0] in user_specific_content_id]
    



def process_content_ids_for_user(cursor, user_data, content_data , content_clustered):
    user_content_scores = []
    
    if len(user_data) > 2:
        if b'0' in content_clustered.keys():
            non_local_ids=content_clustered[b'0']
        else: non_local_ids=[]

        content_data_new=get_pool(content_data,user_data,content_clustered)
        if len(non_local_ids)!=0:
            temp=[(content_data[i][0], content_data[i][1],content_data[i][2],content_data[i][3] ,content_data[i][4]) for i in range(len(content)) if content_data[i][0] in non_local_ids] 
            content_data_new+=temp
        content_data=content_data_new
    
    content_count = len(content_data)

    for j in range(content_count):
        content_score = calc_sim(user_data[1], content_data[j][1])

        loc_score = 0.0
        if len(user_data) > 2:
            loc_score = calc_sim2(user_data[2], content_data[j][2])

        sim_score = None
        if loc_score > 0:
            sim_score = (content_score + loc_score) / 2.0
        else:
            sim_score = content_score


        content_id = content_data[j][0]
        content_modified_timestamp = str(content_data[j][4])
        cos_score_val = sim_score.item()
        if cos_score_val != cos_score_val:
            cos_score_val = 0
        user_content_scores.append((user_data[0], content_id, cos_score_val))

    insert_values = ",".join([str(i) for i in user_content_scores])

    query = "INSERT INTO recommendations (user_id, content_id, cos_score) " \
            "VALUES %s ON DUPLICATE KEY UPDATE cos_score=VALUES(cos_score)" \
            % (insert_values)
    #print('\n\n insert_values',insert_values)
    if len(insert_values)!=0:
        cursor.execute(query)

def update_process_content_ids(db, content_data):
    content_length = len(content_data)
    for i in range(content_length):
        s = "UPDATE content_embedding SET rec_stats=%s WHERE content_id=%s"
        params = (1, content_data[i][0])
        db.cursor.execute(s, params)
    db.conn.commit()


def get_db_conn():
    conn = ML.ML_DB()
    conn.connect_to_db(host='Credentials')
    s = 'use story_sorted'
    cursor = conn.cursor
    cursor.execute(s)

    return conn


def fetch_new_content(db, mod_value):
    cor = db.cursor

    logger.info("Fetching new content...")
    s = "select content_id,content_embedding_1,content_embedding_2,primary_category,content_modified_timestamp from content_embedding where rec_stats=0"
    cor.execute(s)
    content = cor.fetchall()

    logger.info("Content fetched, preprocessing...")

    content = preprocess(content, mod_value)

    content_clustered={}
    for i in range(len(content)):
        if content[i][3] in content_clustered:
            content_clustered[content[i][3]].append(content[i][0])
        else:
            content_clustered[content[i][3]] = [content[i][0]]

    logger.info("Preprocessed, returned %s rows" % (len(content)))
    return content , content_clustered


def process_new_content(db, content,content_clustered):
    cor = db.cursor
    content_count = len(content)

    logger.info("Fetching userdata...")
    s = "select user_id , user_embedding_1 , user_embedding_2 , profile_city , profile_state , e_profile_city , e_profile_state from user_embedding;"
    cor.execute(s)
    user = cor.fetchall()

    logger.info("Fetched %s users" % (len(user)))

    user1, user2 = pre(user)

    logger.info("Adding %s new items for %s users" % (str(content_count), str(len(user))))

    del user

    for i in tqdm(range(len(user1))):
        process_content_ids_for_user(cursor=cor, user_data=user1[i], content_data=content ,content_clustered= content_clustered)
        db.conn.commit()

    logger.info('Processing users with location info, count: %s' % len(user2))

    if len(user2)!=0:
        for i in tqdm(range(len(user2))):
            process_content_ids_for_user(cursor=cor, user_data=user2[i], content_data=content , content_clustered =content_clustered)
            db.conn.commit()

def push_to_slack(message):
    payload = {
        "text": "RecoWriter failed: " + message + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z%z"))}
    requests.post("Credentials")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        logger.error("Usage: python script.py <mod_value>")
        sys.exit()

    mod_value = int(sys.argv[1])

    init_logger(mod_value)
    while True:
        try:
            db = get_db_conn()
            content, content_clustered = fetch_new_content(db, mod_value)

            if len(content) > 0:
                logger.info("Processing %s content pieces" % str(len(content)))
                process_new_content(db, content, content_clustered)
                update_process_content_ids(db, content)
            db.close_connection()
        except Exception as e:
            logger.exception("RecoWriter exception")
            push_to_slack(str(e))

        time.sleep(500)
