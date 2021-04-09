import torch
import sys
from tqdm import tqdm
import mysql.connector
import io
import time
import ML
import numpy
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertModel
import warnings; warnings.simplefilter('ignore')
    
def segm_id(tokenized_text):
    '''
    Creating segement IDs , refer to pytorch pre-trained bert model
    '''
    z=0
    segments_ids=[0,0]
    segments_ids=[]
    for a in tokenized_text:
        if (a == '|'):
            z=1
        segments_ids.append(z)
    return segments_ids

def average_it(encoded_layers,length):
    '''
    Averaging the layers of BERT architecture for embedding generation
    '''
    sum=0
    a=1
    for layer in encoded_layers:
        if(a>9):
            sum=sum+layer
        a+=1
    sum=sum/3
    sum=sum.squeeze()
    sum=torch.sum(sum,dim=0)
    sum=sum/length
    return sum

def bt_Vec(model,text):
    '''
    text=input text in string format to calculate the BERT embedding
    return=feature vector
    '''
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_ids=segm_id(tokenized_text)
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    model.to('cuda')
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    feature=average_it(encoded_layers,len(tokens_tensor))
    return feature

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
    out = io.BytesIO(blob)
    out.seek(0)

    return numpy.load(out)

def clean(text):
    string=""
    for char in text:
        if char.isalpha():
            string=string+char
    return string

def search(city,loc_info):
    a=loc_info[loc_info.city==str(city)].state.values
    if len(a)>0:
        return a[0]
    a=loc_info[loc_info.town==str(city)].state.values
    if len(a)>0:
        return a[0]
    else: return city

def get_db_conn():
    conn = ML.ML_DB()
    conn.connect_to_db(host="credentials")
    s = 'use story_sorted'
    cursor = conn.cursor
    cursor.execute(s)
    return conn

def fetch_user_loc_vec(model,user_meta,loc_info):
    user_loc_vec={}
     
    for i in tqdm(range(len(user_meta))):
        flag=0
        uid=user_meta.iloc[i,0]
        epl=user_meta.iloc[i,1]   
        pl=user_meta.iloc[i,2]
        if type(epl) == str:
            if type(pl) == str:
                epl,pl=clean(epl),clean(pl)
                st_epl,st_pl=search(epl,loc_info),search(pl,loc_info)
                flag=1
        
        else:
            region,city=user_meta.iloc[i,3:]
            if type(region) == str:
                if type(city) == str:
                    flag,epl,st_epl,pl,st_pl=1,city,region,city,region
        if flag==1:            
            v_sum=0
            s_vec=bt_Vec(model,st_epl+' '+ st_epl)
            s_vec=(s_vec+bt_Vec(model,st_pl+' '+ st_pl))/2
            c_vec=bt_Vec(model,epl+' '+ epl)
            c_vec=(c_vec+bt_Vec(model,pl+' '+ pl))/2
            user_loc_v=torch.cat((s_vec,c_vec))
            user_loc_vec[uid]=(user_loc_v.cpu(),pl,st_pl,epl,st_epl)

    return user_loc_vec

def update_user_loc(db,user_loc_vec):

    for uid,val in tqdm(user_loc_vec.items()):
        user_l=adapt_array(val[0])
        s="UPDATE user_embedding SET user_embedding_2=%s ,profile_city=%s,profile_state=%s,e_profile_city=%s,e_profile_state=%s WHERE user_id=%s"

        params=(user_l,val[1],val[2],val[3],val[4],uid)
        db.cursor.execute(s,params)
        
    db.conn.commit()

if __name__ == "__main__":

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased')
    model.eval()

    print("starting process")

    user_meta = pd.read_csv("/home/ubuntu/recSysDB/rawS3data/user_meta",error_bad_lines=False)
    user_meta=user_meta.drop(columns=['first_launch_datetime','old_dbid','country','brand','model'])

    print("loading-loc-info")

    loc_info = pd.read_csv('/home/ubuntu/recSysDB/stateCityData/mapping_state_city.csv')
    
    print("data-loaded")

    user_loc_vec = fetch_user_loc_vec(model,user_meta,loc_info)

    db = get_db_conn()

    update_user_loc(db,user_loc_vec)

    db.close_connection()

