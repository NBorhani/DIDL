# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 06:38:13 2019
General Theta for DN data
@author: ASUS
"""
import numpy as np
from os.path import join
import csv
import xlwt
from tempfile import TemporaryFile

class importdata(object):
    def __init__(self,fileAdress,fileName):
        self.interaction_list, self.shape = self.get_data(fileAdress,fileName)
        #self.train, self.test = self.getTrainTest
        #self.trainDict = self.getTrainDict()
        
    def get_data(self,fileAdress,fileName):
        dict_Rmir2p={}     
        mir_data,pr_data={},{}
        mir_seq,pr_seq={},{}
        with open(join(fileAdress,fileName),'rt', encoding='utf-8') as f:
            f.readline()
            lines = csv.reader(f)
            print(lines)
            i=0
            for line in lines:
                dict_Rmir2p[i] =line[0:]
                i=i+1   
        num_R=len(dict_Rmir2p)
        
        for i in range(num_R):
            mir_data[i]=[dict_Rmir2p[i][0]]
            pr_data[i]=[dict_Rmir2p[i][1:]]
        
        mir_data2,pr_data2={},{}
        mir_data3,pr_data3={},{}
        new_list=[]
        k=0
        for i in range(num_R):
            TF=mir_data[i][0]
            for j in range(len(pr_data[i][0])):
                targets=pr_data[i][0][j]
                if targets=='':
                    continue
                else:
                    new_list.append((TF,targets))
                    mir_data2[k]=[TF]
                    pr_data2[k]=[targets]
                    mir_data3[k]=TF
                    pr_data3[k]=targets
                    k=k+1
                             
            # id sazi
        prots = set(e for val1 in pr_data2.values() for e in val1)
        self.prots2id = {val: ID for ID, val in enumerate(prots)}
        miR = set(e for val1 in mir_data2.values() for e in val1)
        self.miR2id = {val: ID for ID, val in enumerate(miR)}    
        self.shape=[len(self.miR2id),len(self.prots2id)]
            #list int
       
        num_R2=len(new_list)
        interaction_list=[]
        self.mir_seq=mir_seq
        self.pr_seq=pr_seq
        self.mir_data =mir_data
        self.pr_data =pr_data
        
        for i in range(num_R2):
            interaction_list.append((self.miR2id[mir_data3[i]],self.prots2id[pr_data3[i]],1))
        interaction_list=sorted(interaction_list,key=lambda x:(x[0]))
        #interaction_list=sorted(interaction_list,key=lambda x:(x[2]))
        self.maxRate = 1
        return interaction_list,self.shape

    def get_embedding(self,trainlist):  #making matrix
        train_matrix=np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
        for i in range(len(trainlist)):
            mir = trainlist[i][0]
            pr  = trainlist[i][1]
            interaction = trainlist[i][2]
            train_matrix[mir][pr] = interaction
        return np.array(train_matrix)
       

    def get_instances(self, trainlist):
        mir = []
        pr = []
        interaction = []
        for i in trainlist:
            mir.append(i[0])
            pr.append(i[1])
            interaction.append(i[2])
        return np.array(mir), np.array(pr), np.array(interaction)
         
            
    
    
    def decoding_train(self,list_pred,name):
        decode_list=[]
        self.id2miR  = {v: k for k, v in self.miR2id.items()}
        self.id2prots = {v: k for k, v in self.prots2id.items()}
        for i in range(len(list_pred)):
            mir=self.id2miR[list_pred[i][0]]
            mrna=self.id2prots[list_pred[i][1]]
            true_value=list_pred[i][2]
            decode_list.append((mir,mrna,list_pred[i][3],true_value))
        book = xlwt.Workbook()
        sheet1 = book.add_sheet('sheet1')
        
        for i,e in enumerate(decode_list):
            #print(i,e)
            #sheet1.write(i,1,e)
            sheet1.write(i,0,e[0])
            sheet1.write(i,1,e[1])
            sheet1.write(i,2,float(e[2]))
            sheet1.write(i,3,e[3])
                
        #name = "prediction new 1.xls"
        book.save(name)
        book.save(TemporaryFile())
        return decode_list
        

    def name_decoder(self,mir_number,prot_number):
        self.id2miR  = {v: k for k, v in self.miR2id.items()}
        self.id2prots = {v: k for k, v in self.prots2id.items()}
        
        mir_name = self.id2miR[mir_number]
        prot_name = self.id2prots[prot_number]   
        return mir_name,prot_name
    
    
    def excel_saver(self,table_data,excel_file_path):
        import pandas as pd

        # Create a DataFrame
        df = pd.DataFrame(table_data, columns=[self.id2prots[i] for i in range(len(self.id2prots))])
        df.insert(0, 'Name', [self.id2miR[i] for i in range(len(self.id2miR))])

        # Save to Excel
        df.to_excel(excel_file_path, index=False)
        
  
# =============================================================================
#     def excel_saver(self,table_data,file_name):
#         import pandas as pd
#         import os
#         # Specify the directory
#         results_directory = 'RESULTS/Run 3/'
#         # Make sure the directory exists
#         os.makedirs(results_directory, exist_ok=True)
#         # Save to Excel in the specified directory
#         excel_file_path = os.path.join(results_directory, file_name)
#         
#         # Create a DataFrame
#         df = pd.DataFrame(table_data, columns=[self.id2prots[i] for i in range(len(self.id2prots))])
#         df.insert(0, 'Name', [self.id2miR[i] for i in range(len(self.id2miR))])
# 
#         # Save to Excel
#         df.to_excel(excel_file_path, index=False)        
# =============================================================================
  


def generate_interaction_triples(matrix,neg_sampling=1):
    list_of_mat,shuffled_list3=[],[]
    shape=matrix.shape
    # maling list from matrix
    for i in range(shape[0]):
        for j in range(shape[1]):
            list_of_mat.append((i,j,matrix[i,j]))
            
        
    #sorting 1 and 0   
    sort_list=sorted(list_of_mat,key=lambda x:(x[2]))   
    length=len(sort_list)
    num_1=0
    for j in range(length):
        if sort_list[j][2]==1:
            num_1=num_1+1
    num_0=length-num_1     
    sort_list_edge=sort_list[num_0:length]   
    sort_list_NOedge=sort_list[0:num_0] 
    shuffled_listNO = np.random.permutation(sort_list_NOedge)
    shuffled_listNO_modify=[]
    for i in range(len(shuffled_listNO)):
        shuffled_listNO_modify.append((int(shuffled_listNO[i,0]),int(shuffled_listNO[i,1]),shuffled_listNO[i,2]))     
    NO_List=shuffled_listNO_modify[0:int(len(sort_list_edge)*neg_sampling)]
        
    Totall_list=NO_List+sort_list_edge   
    
    # shufel list
    shuffled_list = np.random.permutation(Totall_list)
    for i in range(len(shuffled_list)):
        shuffled_list3.append((int(shuffled_list[i,0]),int(shuffled_list[i,1]),shuffled_list[i,2])) 
        
    return shuffled_list3
