
#!/usr/local/bin/python3

import pandas as pd
import numpy as np
import torch
import sys
from tqdm import tqdm
import mysql.connector
import os
import io
import time
import ML
import numpy
from pytorch_pretrained_bert import BertTokenizer, BertModel


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

def get_db_conn():
    conn = ML.ML_DB()
    conn.connect_to_db(host='Credentials')

    return conn

def preprocess_content_meta():
    data = pd.read_csv("/home/ubuntu/recSysDB/rawS3data/content_meta.csv",error_bad_lines=False,usecols = ['content_title','root_primary_category','published_date','content_id','app','originating_location','primary_category','primary_category_tree','secondary_category_list','published_datetime','modified_datetime'])
    print('New content meta file succesfully loaded \n')
    data = data[data.app.eq("DB")]
    del data['app']
    data=data.dropna(subset=['primary_category_tree'])
    data[['secondary_category_list']] = data[['secondary_category_list']].fillna(value=0)
    data=data.fillna(0)
    data=data.reset_index().drop(columns='index')

    return data

def check_new_content(db):

    s="select distinct content_id from content_embedding"
    db.cursor.execute(s)
    cids=db.cursor.fetchall()

    cids=[cids[i][0] for i in range(len(cids))]
    cids_new=[data.iloc[i,0] for i in range(len(data))]

    if len([value for value in cids if value in cids_new])==len(cids_new):
        db.conn.commit()
        db.close_connection()
        sys.exit("same")

def extract_content_vector(model,data):
    con_con_vec={}

    for i in tqdm(range(len(data))):
        b=0
        id=data.iloc[i,0]
        content=data.iloc[i,1]
        pc=data.iloc[i,6]
        sec=data.iloc[i,7]
        dt_tym=data.iloc[i,3]
        modf_tym=data.iloc[i,5]

        if sec==0:
            b=1
        v_sum=0
        pc_vec=bt_Vec(model,pc+' '+pc)
        v_sum=pc_vec

        if b!=1:
            sec_vec=bt_Vec(model,sec+' '+sec)
            v_sum=pc_vec+sec_vec
        title_vec=(bt_Vec(model,content+' '+content))/2
        v_sum=torch.cat((v_sum,title_vec))
        con_con_vec[id]=(v_sum.cpu().numpy(),dt_tym,modf_tym)

    return con_con_vec

def extract_location_vector(model,data):
    cont_loc_vec={}

    for i in tqdm(range(len(data))):
        b=0
        ci = data.iloc[i,0]
        pct = data.iloc[i,9]
        primary_cat = data.iloc[i,6]
        a = (pct.split('/'))

        if len(a)==5:
            b=1

        v_sum=0

        if len(a)>4:
            if a[1]=='local':
                state=a[2]
                st=state

                for i in range(3,len(a)-2):
                    st=st+' '+a[i]
                    vec=bt_Vec(model,st)  ##dividing by number of words is already done
                    v_sum=v_sum+vec

                v_sum=v_sum/len(st.split())
                ##append v_sum and state and return 1536 dim vec as a content location vec
                state_vec=bt_Vec(model,state+''+state)

                if b==1:
                    con_loc_v=torch.cat((state_vec,state_vec))

                if b==0:
                    con_loc_v=torch.cat((state_vec,v_sum))
                cont_loc_vec[ci]=(con_loc_v.cpu().numpy(),primary_cat)

        b=0

    return cont_loc_vec

def update_content_emb(db,con_con_vec,cont_loc_vec):
    s='use story_sorted'
    db.cursor.execute(s)

    for cid,emb_c in con_con_vec.items():
        cont=adapt_array(emb_c[0])
        dt_tym=emb_c[1]
        modf_tym=emb_c[2]

        if cid in cont_loc_vec.keys():
            loc=adapt_array(cont_loc_vec[cid][0])
            primary_category=cont_loc_vec[cid][1]
            a=1
        else:
            loc=0
            a=0
            primary_category=str(0)
        print('\n New')
        if a==1:
            # local content
            query="SELECT content_id,content_embedding_1,content_embedding_2,primary_category,category,content_creation_timestamp FROM content_embedding WHERE content_id = "+str(cid.item())
            db.cursor.execute(query)
            content_fetched=db.cursor.fetchall()
            if len(content_fetched)!=0:
                print("Old local-content in it")

                # checking dates
                date_fetched = str(content_fetched[0][5])
                date=''
                for val in range(10):
                    date+=str(date_fetched[val])
                if date not in modf_tym:
                    # re-enter
                    print('date', date,  'modf_tym', modf_tym)
                    s = "INSERT INTO content_embedding (content_id,content_embedding_1,content_embedding_2,content_creation_timestamp,category,primary_category,content_modified_timestamp) VALUES(%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE content_embedding_1=%s , content_embedding_2=%s, content_creation_timestamp=%s, category=%s ,rec_stats=%s ,primary_category=%s, content_modified_timestamp=%s"
                    params = (cid.item(), cont, loc, dt_tym, a, primary_category, modf_tym, cont, loc, dt_tym, a, 0, primary_category, modf_tym)
                    db.cursor.execute(s, params)
            else :
                # Not present in the content_embedding
                print("New local item")
                s="INSERT INTO content_embedding (content_id,content_embedding_1,content_embedding_2,content_creation_timestamp,category,primary_category,content_modified_timestamp) VALUES(%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE content_embedding_1=%s , content_embedding_2=%s, content_creation_timestamp=%s, category=%s ,rec_stats=%s ,primary_category=%s, content_modified_timestamp=%s"
                params=(cid.item(), cont, loc, dt_tym, a, primary_category,modf_tym, cont, loc, dt_tym, a , 0 , primary_category ,modf_tym)
                db.cursor.execute(s,params)



        else :
            print(" Not a local content ")
            print("All new items")
            s="INSERT INTO content_embedding (content_id,content_embedding_1,content_embedding_2,content_creation_timestamp,category,primary_category,content_modified_timestamp) VALUES(%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE content_embedding_1=%s , content_embedding_2=%s, content_creation_timestamp=%s, category=%s ,rec_stats=%s ,primary_category=%s, content_modified_timestamp=%s"
            params=(cid.item(), cont, loc, dt_tym, a, primary_category,modf_tym, cont, loc, dt_tym, a , 0 , primary_category ,modf_tym)
            db.cursor.execute(s,params)

    db.conn.commit()
    db.close_connection()

if __name__ == "__main__":

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained('bert-base-multilingual-uncased')
    model.eval()

    data = preprocess_content_meta()

    db = get_db_conn()

    check_new_content(db)

    #Dictionary creation
    cont_con_cocate_dic={}

    print("Training Started")

    con_con_vec = extract_content_vector(model,data)

    print("Titles information extration done")

    print("Extacting location information ")

    cont_loc_vec = extract_location_vector(model,data)

    print("Location information extration done")

    update_content_emb(db,con_con_vec,cont_loc_vec)

    #con_con_vec1 -> dictionary for day or week
    con_con_vec1= np.load('/home/ubuntu/recSysDB/rawS3data/con_con_vec.npy',allow_pickle='TRUE').item()
    cont_loc_vec1= np.load('/home/ubuntu/recSysDB/rawS3data/cont_loc_vec.npy',allow_pickle='TRUE').item()

    #merging
    con_con_vec1 = {**con_con_vec1, **con_con_vec}
    cont_loc_vec1= {**cont_loc_vec1, **cont_loc_vec}
    np.save('/home/ubuntu/recSysDB/rawS3data/con_con_vec.npy',con_con_vec1)
    np.save('/home/ubuntu/recSysDB/rawS3data/cont_loc_vec.npy',cont_loc_vec1)
