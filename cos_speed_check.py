
while(1):
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
    import os
    import csv
    import os
    from tqdm import tqdm
    from numpy import dot
    from numpy.linalg import norm
    import time


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
        if blob==b'0':
            return 0
        out = io.BytesIO(blob)
        out.seek(0)

        return numpy.load(out)

    def calc_sim(vec1,vec2):
        if type(vec1)==int or type(vec2)==int:
            return 0
        return dot(vec1,vec2)/(norm(vec1)*norm(vec2))

    def preprocess(content):
        return [(content[i][0],convert_array(content[i][1]),convert_array(content[i][2])) for i in range(len(content)) if content[i][0] not in cids]

    def pre(user):
        new_none=[]
        new_not_none=[]
        for i in tqdm(range(len(user))):
            if user[i][2]==None:
                new_none.append((user[i][0],convert_array(user[i][1])))
            else:
                new_not_none.append((user[i][0],convert_array(user[i][1]),convert_array(user[i][2])))
        return new_none,new_not_none



    main=ML.ML_DB()
    mydb,cor=main.connect_to_db(host="Credentials")
    s='use story_sorted'
    cor.execute(s)

    #Auto-check
    s="select distinct content_id from recommendations"
    cor.execute(s)
    cids=cor.fetchall()
    cids=[cids[i][0] for i in range(len(cids))]
    print(len(cids))

    s="select content_id,content_embedding_1,content_embedding_2 from content_embedding "
    cor.execute(s)
    content=cor.fetchall()
    content=preprocess(content)
    c_len=len(content)

    if len(content)==0:
        mydb.commit()
        main.close_connection()
        print('same')
        time.sleep(400)
        os.execl(sys.executable, sys.executable, *sys.argv)

    s="select user_id,user_embedding_1,user_embedding_2 from user_embedding "
    cor.execute(s)
    user=cor.fetchall()
    user1,user2=pre(user)
    print("Adding"+str(c_len)+"New items for "+str(len(user))+" users")
    del user

    
    for i in tqdm(range(len(user1))):
        for j in range(c_len):
            cos_score=calc_sim(user1[i][1],content[j][1])
            s="INSERT INTO recommendations (user_id, content_id, cos_score) VALUES(%s,%s,%s)"
            b=(user1[i][0],content[j][0],cos_score.item())
            #cor.execute(s,b)

    for i in tqdm(range(len(user2))):
        for j in range(c_len):
            cos_score=(calc_sim(user2[i][1],content[j][1])+calc_sim(user2[i][2],content[j][2]))/2
            s="INSERT INTO recommendations (user_id, content_id, cos_score) VALUES(%s,%s,%s)"
            b=(user2[i][0],content[j][0],cos_score.item())
            #cor.execute(s,b)            
    
    
    mydb.commit()
    main.close_connection()
    import time
    time.sleep(600)


