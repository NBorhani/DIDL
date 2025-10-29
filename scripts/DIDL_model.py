# Building model
import torch
import torch.nn as nn
import torch.nn.functional as F

class DIDL(nn.Module):
    def __init__(self, matrix, decoder_type = 'cosin', dropout=0.5, 
                         mir_enc =[64,32,25] , prot_enc=[64,32,25]):
        super(DIDL, self).__init__()

        #print('DIDL Loaded')
        self.matrix = matrix
        self.shape = matrix.shape
        self.mir_enc = mir_enc
        self.prot_enc = prot_enc
        self.dropout = dropout
        self.decoder_type = decoder_type
     
        
        self.mir_encoder = nn.Sequential(
                 nn.Linear(self.shape[1], self.mir_enc[0]),
                 #nn.ReLU(inplace=True),
                 #nn.Dropout(0.1),
                 
                 nn.Linear(self.mir_enc[0], self.mir_enc[1]),
                 nn.ReLU(inplace=True),
                 nn.Dropout(self.dropout),
                 
                 nn.Linear(self.mir_enc[1], self.mir_enc[2]),
                 nn.ReLU(inplace=True),
                 )

        self.prot_encoder = nn.Sequential(
                 nn.Linear(self.shape[0], self.prot_enc[0]),
                 #nn.ReLU(inplace=True),
                 #nn.Dropout(0.1),
                 
                 nn.Linear(self.prot_enc[0], self.prot_enc[1]),
                 nn.ReLU(inplace=True),
                 nn.Dropout(self.dropout),
                 
                 nn.Linear(self.prot_enc[1], self.prot_enc[2]),
                 nn.ReLU(inplace=True),
                 ) 
        
        # decoder
        self.Dtensor = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.mir_enc[2], self.prot_enc[2])))


    def forward(self, mir, prot):
        
        #print("here-----",  self.matrix[mir,prot])
        mir_feat  = self.matrix[mir,:]
        prot_feat = torch.transpose(self.matrix[:,prot], 0, 1)
# =============================================================================
#         print('mir_feat', mir_feat.shape)
#         print('prot_feat',prot_feat.shape)
# =============================================================================
        
        mir_out = self.mir_encoder(mir_feat).view(-1, self.mir_enc[2])
        prot_out = self.prot_encoder(prot_feat).view(-1, self.prot_enc[2])
               
# =============================================================================
#         print('mir_out', mir_out.shape)
#         print('prot_out',prot_out.shape)
# =============================================================================
        
        if self.decoder_type == 'cosin':
            interaction_score = F.cosine_similarity(mir_out, prot_out)
            #print(interaction_score)
        elif self.decoder_type == 'Dtensor-sigmoid':     
            mul = 100*torch.matmul(torch.matmul(mir_out, self.Dtensor), prot_out.t())
            decoder_out = torch.sigmoid(mul)
            interaction_score = torch.diagonal(decoder_out, dim1=0, dim2=1)
               
        return interaction_score













# =============================================================================
# input1 = torch.randn(1, 128)
# input2 = torch.randn(1, 128)
# output = F.cosine_similarity(input1, input2)
# print(output.shape)
# 
# torch.reshape(mir_out, (1,25))
# torch.reshape(prot_out, (1,25))
# 
# reshaped_mir_out = mir_out.view(1, 25)
# reshaped_prot_out= prot_out.view(1, 25)
# 
# output = F.cosine_similarity(reshaped_mir_out, reshaped_prot_out)
# print(output.shape)
# =============================================================================
