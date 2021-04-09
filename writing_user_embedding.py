import pandas as pd
import numpy as np
import torch
import sys
from tqdm import tqdm
import mysql.connector
import io
import time
import ML
import numpy
import csv
import os
from tqdm import tqdm

class uci_loader():
    def __init__(self):
        import os
        self.path="/mnt/prod_uci/check/"
        self.a=os.listdir(self.path)
        self.length=len(self.a)
        self.iter=0
        self.uci_data=None
    def _next(self):
        print("loading",self.a[self.iter])
        self.uci_data=pd.read_csv("/mnt/prod_uci/check/"+self.a[self.iter],usecols=['date','user_id','content_id','article_opened','article_scrolled_half','article_scrolled_end','articles_shared'])
        self.iter+=1
        return self.uci_data
    
#u2 are content ids on which we want our UPDATE
def user_update():
    user_content_vectors2={}
    load=uci_loader()
    for l in range(load.length):
        uci_data=load._next()
        uci_data=uci_filtering(uci_data)

        for user_id in (u2):
            user_id=int(user_id)
            c_new=uci_data[uci_data['user_id']==user_id].content_id.values
            s_new=uci_data[uci_data['user_id']==user_id].Score1.values.astype(numpy.int64)
            sum=0
            for i in range(len(s_new)):
                if c_new[i] in con_con_vec.keys():
                    sum=sum+(s_new[i]*(con_con_vec[c_new[i]][0]))
            emb_new=sum/np.sum(s_new)
            if np.sum(s_new)==0:
                emb_new=0

            if l==0:
                emb_old , s_old , c_old = user_content_vectors[user_id]
            else:
                emb_old , s_old , c_old = user_content_vectors2[user_id]
                
            user_c=(((np.sum(s_new)*emb_new)+(s_old*emb_old))/(np.sum(s_new)+s_old))
            c=c_old+len(c_new)
            sc=s_old+np.sum(s_new)
            user_content_vectors2[user_id]=(user_c,sc,c)
    return user_content_vectors2

def uci_filtering(uci):
    uci=uci[uci['content_id'].isin(list(con_con_vec.keys()))]
    uci=uci.dropna()
    s1=uci['article_opened']
    s2=uci['article_scrolled_half']
    s3=uci['article_scrolled_end']
    s4=uci['articles_shared']
    uci['article_opened']=s1/s1
    uci['article_scrolled_half']=s2/s2
    uci['article_scrolled_end']=s3/s3
    uci['articles_shared']=s4/s4
    uci=uci.fillna(0)
    s1=uci['article_opened']
    s2=uci['article_scrolled_half']
    s3=uci['article_scrolled_end']
    s4=uci['articles_shared']
    uci['Score1']=s1*1+s2*2+s3*3+s4*4
    uci=uci[uci.Score1!=0]
    return uci

#content-vectors loading and UCI preprocessing

con_con_vec= np.load('/home/ubuntu/recSysDB/rawS3data/con_con_vec.npy',allow_pickle='TRUE').item()
print('Content-vectors loaded successfully')
uci= pd.read_csv("/home/ubuntu/recSysDB/rawS3data/uci_meta_5_6_20",usecols=['date','user_id','content_id','article_opened','article_scrolled_half','article_scrolled_end','articles_shared'])

uci=uci_filtering(uci)
user_content_vectors={}

##Training Loop

for user_id in tqdm(uci.user_id.unique()):
    user_id=int(user_id)
    c=uci[uci['user_id']==user_id].content_id.values
    s=uci[uci['user_id']==user_id].Score1.values.astype(numpy.int64)
    sum=0
    for i in range(len(s)):
        if c[i] in con_con_vec.keys():
            sum=sum+(s[i]*(con_con_vec[c[i]][0]))
    sum=sum/np.sum(s)
    user_content_vectors[user_id]=(sum,np.sum(s),len(c))

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


main=ML.ML_DB()
mydb,cor=main.connect_to_db(host='credentials')

s='use story_sorted'
cor.execute(s)

s="select user_id,user_embedding_1,score_total,articles_no from user_embedding where articles_no < 24 "
cor.execute(s)
u1=cor.fetchall()

s="select user_id from user_embedding where articles_no > 24 "
cor.execute(s)
u2=cor.fetchall()
u2=[u2[i][0] for i in range(len(u2)) if u2[i][0] in user_content_vectors.keys()]


#For Merging

for uid,emb_old,sc_old,c_old in tqdm([u1[i] for i in range(len(u1))]):
    if uid in user_content_vectors.keys():
        emb_nw,sc_new,c_new=user_content_vectors[uid]
        emb_old=convert_array(emb_old)
        
        user_c=adapt_array(((sc_new*emb_nw)+(sc_old*emb_old))/(sc_new+sc_old))
        c=c_old+c_new
        sc=sc_old+sc_new
        
        s="INSERT INTO user_embedding (user_id,user_embedding_1,articles_no,score_total) VALUES(%s,%s,%s,%s) ON DUPLICATE KEY UPDATE user_embedding_1=%s , articles_no=%s, score_total=%s"
        
        b=(uid, user_c, c, sc.item(), user_c, c, sc.item())
        cor.execute(s,b)

# straight update  
user_content_vect2=user_update()
for uid,emb_u in tqdm(user_content_vect2.items()):
    user_c=adapt_array(emb_u[0])
    sc=emb_u[1]
    c=emb_u[2]
    #Don't need to fetch as directly updating

    s="UPDATE user_embedding SET user_embedding_1=%s , articles_no=%s, score_total=%s WHERE user_id=%s AND articles_no > 24"
    
    b=(user_c, c, sc.item(), uid)
    cor.execute(s,b)    

#i for new users in uci-file    
for uid,emb_u in tqdm(user_content_vectors.items()):
    user_c=adapt_array(emb_u[0])
    sc=emb_u[1]
    c=emb_u[2]

    s="INSERT IGNORE INTO user_embedding (user_id,user_embedding_1,articles_no,score_total) VALUES (%s,%s,%s,%s)"

    b=(uid, user_c, c, sc.item())
    cor.execute(s,b)


    
mydb.commit()
main.close_connection()
        

