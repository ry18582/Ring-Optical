#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:44:05 2019

@author: Lida
"""

#import networkx as nx
import matplotlib as plt
import numpy as np
import pandas as pd
import csv
from itertools import permutations

node_list=[0,1,2,3,4]
Traffic_SD_pairs=list(permutations(node_list,2))

Bandwidth=[1,2,4,16]
Traffic=range(18,26)
traffic=[x*50 for x in Traffic]
Traffic_number=[1]+traffic
#print(Traffic)
for i in range(len(traffic)):
    Traffic_num=Traffic_number[i]
    Traffic=[]
    for i in range(Traffic_num):
        random_seed=np.random.randint(48)
        if(random_seed<12):
            S_band=Bandwidth[0]
        if(random_seed<24 and random_seed>11):
            S_band=Bandwidth[1]
        if(random_seed<36 and random_seed>23):
            S_band=Bandwidth[2]
        if(random_seed<48 and random_seed>35):
            S_band=Bandwidth[3]        
        Traffic.append(Traffic_SD_pairs[np.random.randint(len(Traffic_SD_pairs))]+(S_band,))
        
    filename='traffic_'+str(Traffic_num)+'.csv'
    MyFile=open(filename,'w')
    writer=csv.writer(MyFile,lineterminator='\n')
    for element in Traffic:
        writer.writerow(element)
    MyFile.close()
    
print("END")
