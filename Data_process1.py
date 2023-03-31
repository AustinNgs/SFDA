# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 18:23:32 2020

@author: Daddy Wu
"""
import os
import numpy as np
import pandas as pd
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch
import math
import random

class MyDataset(data.Dataset):
    
    def __init__(self,dataset_dir,idx,data_flag):
        
        cuda = True
        idx_Label=[]
        Data=[]
        Label=[]
        new_idx_Label=[]
        for k in range(2):
            path=os.path.join(dataset_dir,r'fix_%i_三步_berg_40_50patients_overlap2'%(k))
            #path=os.path.join(dataset_dir,r't=19_Layer2all%i'%(k))
            #path=os.path.join(dataset_dir,r'Layer2_%i'%(k))
            #path=os.path.join(dataset_dir,r'new%i'%(k))
            path_list = os.listdir(path)
            
            for filename in path_list:
                
                idx_name=filename.split('_')[1]
                idx_name=idx_name.split('号')[0]
                idx_name=int(idx_name)
                '''
                if idx_name==8:
                    idx_name=7
                if idx_name==9:
                    idx_name=8
                if idx_name==10:
                    idx_name=9
                if idx_name==11:
                    idx_name=10
                if idx_name==12:
                    idx_name=11
                if idx_name==13:
                    idx_name=12
                if idx_name==14:
                    idx_name=13
                if idx_name==15:
                    idx_name=14
                if idx_name==17:
                    idx_name=15
                if idx_name==19:
                    idx_name=16
                if idx_name==20:
                    idx_name=17
                if idx_name==22:
                    idx_name=18
                if idx_name==23:
                    idx_name=19
                if idx_name==24:
                    idx_name=20
                if idx_name==25:
                    idx_name=21
                if idx_name==27:
                    idx_name=22
                if idx_name==29:
                    idx_name=23
                if idx_name==31:
                    idx_name=24
                if idx_name==32:
                    idx_name=25
                if idx_name==34:
                    idx_name=26
                if idx_name==36:
                    idx_name=27
                if idx_name==40:
                    idx_name=28
                if idx_name==41:
                    idx_name=29
                if idx_name==42:
                    idx_name=30
                if idx_name==43:
                    idx_name=31
                if idx_name==44:
                    idx_name=32
                if idx_name==45:
                    idx_name=33
                if idx_name==47:
                    idx_name=34
                if idx_name==48:
                    idx_name=35
                if idx_name==50:
                    idx_name=36
                '''

                if idx_name==49:
                    idx_name=16
                if idx_name==50:
                    idx_name=38

                if idx_name==1 or idx_name==2 or idx_name==3 or idx_name==4:
                    new_idx=0
                if idx_name==6 or idx_name==7 or idx_name==8 or idx_name==10:
                    new_idx=0
                if idx_name==11 or idx_name==14:
                    new_idx=0
                if idx_name==15 or idx_name==20 or idx_name==22 or idx_name==24:
                    new_idx=0
                if idx_name==25 or idx_name==30 or idx_name==33 or idx_name==36:
                    new_idx=0
                if idx_name==35 or idx_name==39 or idx_name==40 or idx_name==41:
                    new_idx=0
                if idx_name==42 or idx_name==43 or idx_name==44 or idx_name==46:
                    new_idx=0
                if idx_name==47 or idx_name==48 or idx_name==16 or idx_name==38:
                    new_idx=0

                if idx_name==5 or idx_name==9 or idx_name==17 or idx_name==18:
                    new_idx=1
                if idx_name==19 or idx_name==21 or idx_name==23 or idx_name==26:
                    new_idx=1
                if idx_name==27 or idx_name==29 or idx_name==32 or idx_name==37 or idx_name==45:
                    new_idx=1
                if idx_name==12 or idx_name==13 or idx_name==28 or idx_name==31 or idx_name==34:
                    new_idx=1


                idx_name=int(int(idx_name)-1)
                new_idx=int(new_idx)
                #print(idx_name)
                df = pd.read_csv(os.path.join(path, filename),encoding='UTF-8')

                sample_y=np.array(df['label'])[0]
                sample_x=df[['L1','L2','L3','L4','L5','L6','L7','L8','R1','R2','R3','R4','R5','R6','R7','R8']]
                #sample_xx=sample_x[::-1]
                sample_x = sample_x.transpose()
                #sample_xx = sample_xx.transpose()
                sample_x=np.array(sample_x)
                #sample_xx=np.array(sample_xx)

                if (sample_x.max(0)-sample_x.min(0)).all()>0:
                    sample_x=(sample_x-sample_x.min(0))/(sample_x.max(0)-sample_x.min(0))
                    sample_x=np.expand_dims(sample_x,axis=0)
                    Data.append(sample_x.tolist())
                    Label.append(sample_y)
                new_idx_Label.append(new_idx)
                idx_Label.append(idx_name)
                
                '''
                if (sample_xx.max(0)-sample_xx.min(0)).all()>0:
                    sample_xx=(sample_xx-sample_xx.min(0))/(sample_xx.max(0)-sample_xx.min(0))
                    sample_xx=np.expand_dims(sample_xx,axis=0)
                    Data.append(sample_xx.tolist())
                    Label.append(sample_y)
                idx_Label.append(idx_name)
                '''       
        Data=np.array(Data)
        Label=np.array(Label) 
        idx_Label=np.array(idx_Label)
        new_idx_Label=np.array(new_idx_Label)

        train_num=[]
        test_num=[]
        change_num=[]
        val_num=[]
        for i in range(len(idx_Label)):
            #print(idx,idx_Label[i],idx_Label.shape)
            if idx_Label[i]==idx:
                test_num.append(i)
            else:
                train_num.append(i)

            if idx_Label[i]==47:
                change_num.append(i)

            random.shuffle(train_num)    
            train_num1=train_num[0:math.floor(len(train_num)*0.8)]
            val_num=train_num[math.floor(len(train_num)*0.8):]

        if data_flag=='train':

            if idx!=47:
                idx_Label[change_num]=idx

            Data=Data[train_num1,:]
            Label=Label[train_num1]
            idx_Label=idx_Label[train_num1]
            new_idx_Label=new_idx_Label[train_num1]
            
            train_ind = np.random.permutation(len(train_num1))
            self.Data=Data[train_ind,:]
            self.Label=Label[train_ind]
            self.idx_Label=idx_Label[train_ind]
            self.new_idx_Label=new_idx_Label[train_ind]
        
        elif data_flag=='val':

            if idx!=47:
                idx_Label[change_num]=idx

            Data=Data[val_num,:]
            Label=Label[val_num]
            idx_Label=idx_Label[val_num]
            new_idx_Label=new_idx_Label[val_num]
            
            val_ind = np.random.permutation(len(val_num))
            self.Data=Data[val_ind,:]
            self.Label=Label[val_ind]
            self.idx_Label=idx_Label[val_ind]
            self.new_idx_Label=new_idx_Label[val_ind]
            
        elif data_flag=='test':
            Data=Data[test_num,:]
            Label=Label[test_num]
            idx_Label=idx_Label[test_num]
            new_idx_Label=new_idx_Label[test_num]
            
            test_ind = np.random.permutation(len(test_num))
            self.Data=Data[test_ind,:]
            self.Label=Label[test_ind]
            self.idx_Label=idx_Label[test_ind]
            self.new_idx_Label=new_idx_Label[test_ind]
            

    def __getitem__(self, item):
        
        
        input_data=self.Data[item,:]
        output_risk=self.Label[item]
        output_domain=self.idx_Label[item]
        output_new=self.new_idx_Label[item]

        return input_data,output_risk,output_domain



    def __len__(self):
        return len(self.Data)
   