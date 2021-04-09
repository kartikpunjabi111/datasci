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
from numpy import dot
from numpy.linalg import norm

main=ML.ML_DB()
mydb,cor=main.connect_to_db(host = 'credentials')

s='use story_sorted'
cor.execute(s)

import datetime
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)
print(yesterday)
s="SELECT content_id FROM content_embedding WHERE updated_timestamp BETWEEN '"+str(yesterday)+" 00:00:00' AND '"+str(yesterday)+" 23:59:59'"
cor.execute(s)
yest_cids=cor.fetchall()
yest_cids=[yest_cids[i][0] for i in range(len(yest_cids))]
yest_cids=tuple(yest_cids)

s="delete from recommendations where content_id in %s"
b=yest_cids
cor.execute(s,b)

#s="delete from content_embedding where content_id in %s"
#b=yest_cids
#cor.execute(s,b)

mydb.commit()
main.close_connection()
