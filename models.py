from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import logging
import helper
import torch.nn.functional as F  
import train
import numpy as np
import torch
from torch import nn
from torch.nn import init
import pandas as pd
import sklearn
class Attention(nn.Module):
    def __init__(self):  
        super(Attention,self).__init__()   
        self.softmax = torch.nn.Softmax()
        self.dropout = nn.Dropout()
    def forward(self,Q,K,V):
        d_k = Q.size()[-1]
        outputs = torch.matmul(Q, K.transpose(-1, -2))  # (N, T_q, T_k)
        outputs /= d_k ** 0.5
        out1 = self.softmax(outputs)
        outputs = self.dropout(outputs)
        context_vectors = torch.matmul(outputs, V)
        return context_vectors


# 创建一个 GCN 模型
class GCNNet(nn.Module):
    def __init__(self,nums_attribute_node = 22,nums_therapy_node = 6,embedding_dim = 16, hidden_dim = 16,negative_length=1,data=None,negative_masks=[1,1]):
        super(GCNNet, self).__init__()
        self.nums_attribute_node=nums_attribute_node
        self.nums_therapy_node=nums_therapy_node
        self.embedding_dim = embedding_dim
        self.embed_attributes = nn.Linear(self.nums_attribute_node,embedding_dim)
        self.embed_therapies = nn.Linear(self.nums_therapy_node,embedding_dim)

        self.conv_layer = GCNConv(embedding_dim, embedding_dim)
        
        self.mlp_for_gnn = nn.Sequential(
        nn.Linear(embedding_dim, hidden_dim), 
        nn.ReLU(), 
        nn.Linear(hidden_dim, 2) 
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.negative_length=negative_length 
        
        self.attention_layer=Attention() 
        self.linear_layer = nn.Linear(2,1)
        
        self.rule=helper.return_rule(data)
        self.data=data
        self.negative_masks=negative_masks
        
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        self.bn4 = nn.BatchNorm1d(embedding_dim)
        self.node_vector_dict= {}
        self.therapy_vector_dict ={}
        
    def get_negative_query(self,node,query,rule_tag_s,batch_data,negative_resource=[1,1]):
        negative_query_sub = [] 
        if negative_resource[0]==1: 
            for theray in batch_data['therapies'].x: # 增加负样本集
                if theray not in query:
                    theray=F.one_hot(theray,num_classes=self.nums_therapy_node).float() 
                    temp = self.embed_therapies(theray) 
                    if len(negative_query_sub) >= self.negative_length:
                        break
                    else:
                        negative_query_sub.append(temp)
        if negative_resource[1]==1:     
            for attribute_neg_node in batch_data['attributes'].x: 
                if len(negative_query_sub) >= self.negative_length:  
                    break
                for rule_item in rule_tag_s:
                    if attribute_neg_node in self.rule[rule_item-1]:
                        break
                
                attribute_neg_node = F.one_hot(attribute_neg_node,num_classes=self.nums_attribute_node).float()#shape为：节点数x属性总数 #38,52
                attribute_neg_node_embed = self.embed_attributes(attribute_neg_node)
                negative_query_sub.append(attribute_neg_node_embed)
                            
        while len(negative_query_sub) < self.negative_length:
            negative_query_sub.append(torch.zeros(self.embedding_dim))
                    
        return negative_query_sub 
      
    def get_loss2(self,list_attr_vec,list_therapy_vec):
        loss2 = 0
        for rule_index in self.data['rules']:
            temp_a=[]
            temp_r=[]
            for i in self.data['attributes'].x:
                if rule_index in self.data['attributes'].rule_tag[int(i)]: 
                    temp_a.append(int(i))
            
            for i in self.data['therapies'].x:
                if rule_index in self.data['therapies'].rule_tag[int(i)]: 
                    temp_r.append(int(i))

            vec_attr = [list_attr_vec[i] for i in temp_a]
            vec_attr = torch.mean(torch.stack(vec_attr),axis = 0)
            vec_therapy = [list_therapy_vec[i]for i in temp_r]
            vec_therapy = torch.mean(torch.stack(vec_therapy),axis = 0)    
            loss2_item = torch.sigmoid(torch.matmul(vec_therapy, vec_attr))
            loss2+=loss2_item
        return loss2
                   
    def forward(self, batch_data,positive_confuse_way='add',fused_weight=[0,1,0]):
        x_a = batch_data['attributes'].x 
        node_vectors = []
        negative_query =[] 
        therapy_vectors =[] 
        attention_vector =[] 
        num_real_positive = []
        
        for node in x_a:
            negative_query_sub = [] 
            rule_tag_set=batch_data['attributes'].rule_tag[int(node)]
            
            query_set = []
            for therapy in batch_data['therapies'].x:  
                for rule_attribute_node in rule_tag_set: 
                    if rule_attribute_node in batch_data['therapies'].rule_tag[int(therapy)]:
                        query_set.append(therapy) 
                        break
            query_set = list(set([int(i) for i in query_set]))    
            
            negative_query_sub =   self.get_negative_query(node,query_set,rule_tag_set,batch_data,self.negative_masks)      
            negative_query.append(negative_query_sub)
            
            key_list = []
            for rule_item in rule_tag_set:
                key=torch.tensor(self.rule[rule_item-1])
                key_list.extend(key)
            key_list = torch.tensor(list(set([int(i) for i in key_list])))
            one_hot = F.one_hot(key_list,num_classes=self.nums_attribute_node).float()
            key_embed = self.embed_attributes(one_hot) 
            
            query_set = torch.tensor(query_set)
            query = F.one_hot(query_set,num_classes=self.nums_therapy_node).float()
            query_embed = self.embed_therapies(query) 

            value_embed = key_embed
            
            node = F.one_hot(node,num_classes=self.nums_attribute_node).float()
            node_vector = self.embed_attributes(node)
            node_reshape = node_vector.unsqueeze(0) 
            
            attention_output = self.attention_layer(query_embed,key_embed,value_embed) 
            attention_output = torch.mean(attention_output, dim=0, keepdim=True) 
            attention_vector.append(attention_output)  
            node_vectors.append(node_reshape)
            
            num_real_positive.append(query_embed.size(0))
            padding_size = 10-query_embed.size(0)
            query_embed = F.pad(query_embed,(0,0,0,padding_size),mode='constant', value=0)
            query_reshape = query_embed.unsqueeze(0)  
            therapy_vectors.append(query_reshape)
         
         
        negative_query_t = []
        t_node_vectors =[] 
        
        for t_node in batch_data['therapies'].x:
            rule_tag_t_set = batch_data['therapies'].rule_tag[int(t_node)]
            same_rules_therapy =[] 
            for therapy in batch_data['therapies'].x:  
                for rule_therapy in rule_tag_t_set: 
                    if rule_therapy in batch_data['therapies'].rule_tag[int(therapy)]: 
                        same_rules_therapy.append(therapy) 
                        break
            same_rules_therapy = list(set([int(i) for i in same_rules_therapy]))    
            negative_query_sub_t =   self.get_negative_query(t_node,same_rules_therapy,rule_tag_t_set,batch_data,[1,0]) #self.negative_masks     
            negative_query_t.append(negative_query_sub_t)
 
            # query 列表 保存的是 条件节点
            query_list = []
            for rule_item in rule_tag_t_set:
                query=torch.tensor(self.rule[rule_item-1])
                query_list.extend(query)
            query_list = torch.tensor(list(set([int(i) for i in query_list])))
            one_hot = F.one_hot(query_list,num_classes=self.nums_attribute_node).float()
            query_embed = self.embed_attributes(one_hot)  
            
            key_set = torch.tensor(same_rules_therapy)
            key = F.one_hot(key_set,num_classes=self.nums_therapy_node).float()
            key_embed = self.embed_therapies(key)

            value_embed = key_embed
            
            t_node = F.one_hot(t_node,num_classes=self.nums_therapy_node).float()
            t_node_vector = self.embed_therapies(t_node) 
            t_node_reshape = t_node_vector.unsqueeze(0) 
            
            
            attention_output = self.attention_layer(query_embed,key_embed,value_embed) 
            attention_output = torch.mean(attention_output, dim=0, keepdim=True) 
            attention_vector.append(attention_output)  
            t_node_vectors.append(t_node_reshape) 

            padding_size = 10-query_embed.size(0)
            query_embed = F.pad(query_embed,(0,0,0,padding_size),mode='constant', value=0)
            query_reshape = query_embed.unsqueeze(0)  
        negative_query=torch.stack([torch.stack(i) for i in negative_query])
        negative_query_t=torch.stack([torch.stack(i) for i in negative_query_t])
        attributes_vectors = torch.cat(node_vectors,dim =0) 
        therapy_vectors = torch.cat(therapy_vectors,dim =0)  
        attention_vector = torch.cat(attention_vector,dim =0) 

        therapy_node_embed = torch.cat(t_node_vectors,dim =0)##
        conv_pred_vector = torch.cat([attributes_vectors,therapy_node_embed],dim = 0)
        edge_pre_for_conv = batch_data['attribute_to_therapies'].edge_index.clone()
        edge_pre_for_conv[1]=edge_pre_for_conv[1]+len(attributes_vectors) 
        conv_vector = self.conv_layer(conv_pred_vector,edge_pre_for_conv) 

        if  positive_confuse_way =='add':
            attributes_mixed = fused_weight[0]*attributes_vectors+fused_weight[1]*attention_vector[:len(attributes_vectors)]+fused_weight[2]*conv_vector[:len(attributes_vectors)]
            therapy_mixed = fused_weight[0]*therapy_node_embed+fused_weight[1]*attention_vector[len(attributes_vectors):]+fused_weight[2]*conv_vector[len(attributes_vectors):]

        attributes_mixed = attributes_mixed.unsqueeze(1)
        therapy_mixed = therapy_mixed.unsqueeze(1)
        
        list_attr_vec=[]
        for node in x_a:
            self.node_vector_dict[int(node)] = attributes_mixed[int(node)][0]
            list_attr_vec.append(attributes_mixed[int(node)][0])
            
        list_therapy_vec=[]
        for therapy in batch_data['therapies'].x: 
            self.therapy_vector_dict[int(therapy)] = therapy_mixed[int(therapy)][0]
            list_therapy_vec.append(therapy_mixed[int(therapy)][0])
    
        positive = attributes_mixed * therapy_vectors 
        logging.info('positive:'.format(positive.shape)) 

        negative = attributes_mixed * negative_query 
        negative_t = therapy_mixed * negative_query_t 
        
        logging.info('negative:%s',negative.shape) 

        result_attribute = torch.cat((positive,negative),dim=1)
        result_therapy = negative_t
        result_therapy = torch.sigmoid(result_therapy)
        result_attribute  = torch.sigmoid(result_attribute)
        
        loss2 = self.get_loss2(list_attr_vec,list_therapy_vec)
        return result_attribute,result_therapy,num_real_positive,loss2
 
class DPCNN_v1(nn.Module):
    def __init__(self, num_filters,num_classes):
        super(DPCNN_v1, self).__init__()
        self.num_filters=num_filters
        self.num_classes = num_classes
        self.embed=300
        self.conv_region = nn.Conv2d(1, self.num_filters, (3, self.embed), stride=1)
        self.conv = nn.Conv2d(self.num_filters, self.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.num_filters, self.num_classes)
        self.nn1=nn.Linear(1,300)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.nn1(x)
        #x = x.view(x.size(0), 1, x.size(1), x.size(2))

        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
    
class Decision_block(nn.Module):
    """_summary_

    Args:
        rule_tunnel (torch.Tensor): 规则通道得到的张量，形状是 N*M,N是样本数目，M是嵌入的维数
        data_tunnel (torch.Tensor): 数据通道得到的张量，形状是 N*M,N是样本数目，M是嵌入的维数
        similarity_tunnel (torch.Tensor): 相似节点聚合通道得到的张量，形状是 N*M,N是样本数目，M是嵌入的维数
        fused_way(str):可以选择的方法，在列表['add','mlp']
    Returns:
        tunnel_vector_fused _type_: 一个向量
    """
    
    def __init__(self, decision_way='add',DataEncoder_Output_Dim=32):
        super(Decision_block, self).__init__()
        self.decision_way = decision_way
        self.fc_out = DataEncoder_Output_Dim
        self.fc = nn.Linear(self.fc_out,self.fc_out)

        self.fc_adapted = nn.Linear(3, 1,bias=False)# 输入维度为3，输出维度为1
        self.fc_adapted_2 = nn.Linear(self.fc_out,self.fc_out)
    def forward(self,rule_tunnel,data_tunnel,similarity_tunnel):
        if self.decision_way == 'add':
            out =  self.add_fused(rule_tunnel,data_tunnel,similarity_tunnel)
        if self.decision_way == 'fc':
            out = self.liner_fused(rule_tunnel,data_tunnel,similarity_tunnel)
        if self.decision_way == 'fc_adapted':
            out = self.adatped_liner_fused(rule_tunnel,data_tunnel,similarity_tunnel)
            return out,self.fc_adapted.weight
        return out,None
    
    def add_fused(self,tunnel_1,tunnel_2,tunnel_3):
        output_matrix = tunnel_1+tunnel_2+tunnel_3
        return output_matrix
    
    def liner_fused(self,tunnel_1,tunnel_2,tunnel_3):
        output_matrix = self.fc( tunnel_1+tunnel_2+tunnel_3) 
        return output_matrix
    
    def adatped_liner_fused(self,tunnel_1,tunnel_2,tunnel_3):
        combined_matrix = torch.stack((tunnel_1,tunnel_2,tunnel_3), dim=2)
        output_matrix = self.fc_adapted(combined_matrix) 
        output_matrix=output_matrix.squeeze()
        return output_matrix
        
# 处理连续数据
class net_continues(nn.Module):
  """处理连续数据的mlp

  Args:
      nn (_type_): _description_
  """
  def __init__(self,input_dim ,hidden_dim,out_dim):
      super(net_continues,self).__init__()
      self.hidden_dim = hidden_dim
      self.out_dim = out_dim
      self.input_dim = input_dim
      self.continue_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim)
        )
  def forward(self,x):
      out = self.continue_net(x)
      return out
  
class DataEncoder(nn.Module):
    def __init__(self,attribute_list, hidden_dim, output_dim,mlp_hidden_dim=16):
        super(DataEncoder, self).__init__()
        self.hidden_dim = hidden_dim 
        
        self.attribute_list = attribute_list
        
        self.mlp_input_dim = hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_output_dim = output_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim*len(self.attribute_list), self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_output_dim)
        )

        self.continuous_attributes_input_dim = 1
        self.continuous_embedding = nn.Sequential(
            nn.Linear(self.continuous_attributes_input_dim, self.hidden_dim) 
        )
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
        
        # dpcnn settings:
        self.conv_region = nn.Conv2d(1, hidden_dim, (3, output_dim), stride=1)
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, patient_info,continuous_attributes, train_way='mlp'):
        """就是训练过程啦

        Args:
            patient_info (list): AVP的列表，但是注意这里输入的是单个患者的AVP列表，而不是批量数据
            continuous_attributes (list): 一个存放了属性名的列表
            train_way (str, optional): 训练方法，也就是数据驱动使用的编码器. Defaults to 'mlp'.

        Raises:
            ValueError: 如果选择的train_way不在候选集内会报错

        Returns:
            output _type_: 输出经过了数据编码器之后的隐藏向量
        """
        # 1、根据attribute_list的长度，创建一个len(attribute_list) * hidden_dim的全零矩阵，叫patient_max。
        patient_max = torch.zeros(len(patient_info),len(self.attribute_list), self.hidden_dim).to(device=self.device)

        # 2、把patient_info转换为len(attribute_list) * hidden_dim的向量
        for info_list in range(len(patient_info)):  #患者的index 
            for info in patient_info[info_list]: #AVP
                attribute, value = info.split(':')
                index = self.attribute_list.index(attribute)
            
                if attribute not in continuous_attributes:
                    patient_max[info_list][index] = self.one_hot_encode(value)
                else:
                    embedded_value = self.continuous_embedding(torch.tensor([float(value)]).to(device=next(self.continuous_embedding.parameters()).device))
                    patient_max[info_list][index] = embedded_value
                    
        # 3、根据train_way选择融合方法
        if train_way == 'mlp':
            patient_max = patient_max.view(( 1,-1))
            output = self.mlp(patient_max)
        elif train_way == 'dpcnn':
            output = self.dpcnn_forward(patient_max) 
        else:
            raise ValueError("Invalid train_way. Choose from ['mlp', 'transformer', 'dpcnn']")

        return output
    
    def  _block(self, x):
        """
            DPCNN组件的一部分 
        """
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
    def dpcnn_forward(self,patient_max):
        x =patient_max# x[0]  # batch,fil_num,badding_dim
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]           
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]  
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]       
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] >= 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        output = x#self.dpcnn(patient_max)
        return output
    def one_hot_encode(self, value):
        one_hot = torch.zeros(self.hidden_dim)
        one_hot[int(value)] = 1
        return one_hot

# 该代码来自
# https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/SelfAttention.py
class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class fused_tunnels_MLP(nn.Module):
    def __init__(self,hidden_size2=16,output_size2=17):
        super(fused_tunnels_MLP, self).__init__()
        self.linear_layer = nn.Linear(3, 1)
        self.linear_layer2 = nn.Linear(hidden_size2, output_size2)
    def forward(self,rule_tunnel,data_tunnel,similarity_tunnel,label_n=17):
        combined_matrix = torch.stack((rule_tunnel,data_tunnel,similarity_tunnel), dim=2)
        output_matrix = self.linear_layer(combined_matrix) 
        output_matrix=output_matrix.squeeze()
        output_matrix = self.linear_layer2(output_matrix)
        return output_matrix
class Three_Tunnels(nn.Module):
    def __init__(self,data_vec,data,Memory_set,attribute_list,target_attribute,DataEncoder_Output_Dim=16,fused_hidden=16,fused_output=7,dataencoder='MLP',decision_way='add'):
        super(Three_Tunnels, self).__init__()
        self.data_vec=data_vec
        self.attribute_vec=data_vec[0] 
        self.therapy_vec=data_vec[1]
        self.data=data 
        self.dataencoder=DataEncoder(attribute_list=attribute_list,hidden_dim=self.data_vec[0][0].shape[0], output_dim=DataEncoder_Output_Dim,mlp_hidden_dim=16)

        self.loss_CrossEntropy = nn.CrossEntropyLoss() 
        self.loss_MSE = nn.MSELoss()
        
        self.target_attribute = target_attribute 
        
        self.Memory_set=Memory_set
        self.Memory_y = Memory_set[self.target_attribute]
        
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
        
        self.bn1=nn.BatchNorm1d(DataEncoder_Output_Dim)
        self.bn2=nn.BatchNorm1d(DataEncoder_Output_Dim)
        self.bn3=nn.BatchNorm1d(DataEncoder_Output_Dim)
        
        self.decision_block = Decision_block(decision_way=decision_way)
        self.fc_hid2classnum = nn.Linear(DataEncoder_Output_Dim, fused_output)
        
    def forward(self,train_AVP_list,Train_set,similiar_num=5,mask_list=[1,1,1],weight1=0.5):
        Train_set_1=Train_set.fillna(-1, inplace=False)
        Memory_set_1=self.Memory_set.fillna(-1, inplace=False) #Memory_set_1 把空值变成了-1
        Memory_set_1.drop(self.target_attribute, axis=1, inplace=True) #Memory_set_1去掉了目标属性
        matched_no_Nodes = 0
        train_fused_vector=[] 
        for i in range(len(train_AVP_list)): 
            attribute_index = train.compare_with_AVP_dict(train_AVP_list[i], self.data['attributes'].AVP_dict,self.data['attributes'].continuous_attributes)
            if len(attribute_index) == 0:
                print(f'patient {i} / {len(train_AVP_list)} has no matched nodes!')
                matched_no_Nodes+=1
            fused_vector = train.fuse_attribute_node(self.attribute_vec, attribute_index, fuze_way='mean')
            train_fused_vector.append(fused_vector)
        vector_matrix = torch.stack(train_fused_vector, dim=0).to(device=self.device)
        if mask_list[1] == 0:
            data_tunnel = torch.zeros_like(vector_matrix)
        else:
            data_tunnel = self.dataencoder(patient_info=train_AVP_list, continuous_attributes=self.data['attributes'].continuous_attributes,train_way='dpcnn')

        if mask_list[2] == 0:
            similarity_tunnel = torch.zeros_like(vector_matrix)
        else:
            similarity_tunnel=[]
            similarity_fused_vector_y = [] #记忆集中 治疗方案的节点表示
            memory_fused_vector=[] #记忆集中属性融合节点
            Memory_list_attribute = helper.create_attribute_lists(Memory_set_1)
            Memory_y_df=pd.DataFrame({self.target_attribute: self.Memory_y})
            simliary_y_list = helper.create_attribute_lists(Memory_y_df)
            
            for i in range(len(Memory_list_attribute)):
                attribute_index = train.compare_with_AVP_dict(Memory_list_attribute[i], self.data['attributes'].AVP_dict,self.data['attributes'].continuous_attributes)
                fused_vector = train.fuse_attribute_node(self.attribute_vec, attribute_index, fuze_way='mean')
                memory_fused_vector.append(fused_vector)
                
                therapy_index = train.compare_with_AVP_dict(simliary_y_list[i], self.data['therapies'].AVP_dict,[])
                fused_vector = train.fuse_attribute_node(self.therapy_vec, therapy_index, fuze_way='mean')
                similarity_fused_vector_y.append(fused_vector)
            
            memory_vector = torch.stack(memory_fused_vector, dim=0).to(device=self.device)
            memory_vector_y = torch.stack(similarity_fused_vector_y, dim=0).to(device=self.device)
            similarity_ = sklearn.metrics.pairwise.cosine_similarity(vector_matrix.cpu().detach().numpy(), memory_vector.cpu().detach().numpy()) 
            top_indices = np.argsort(-similarity_, axis=1)[:, :similiar_num] 
            
            weighted_values = np.zeros((len(top_indices),similiar_num)) 
            for i in range(weighted_values.shape[0]): 
                weighted_values[i] = similarity_[i,top_indices[i]]
                
            for indices_index in range(len(top_indices)):
                indices  = top_indices[indices_index ]
                weight_sample = weighted_values[indices_index]
                weight_sample = helper.softmax(weight_sample)
                Y_k = memory_vector_y[indices]  
                similarity_fused_vector_y = [i for i in Y_k]
                fused_vector_t = train.similary_vector_fused(similarity_fused_vector_y,fuse_way='WeightedAdd',similiar_num=similiar_num,weighted_values=weight_sample)
                similarity_tunnel.append(fused_vector_t)
            
            similarity_tunnel = torch.stack(similarity_tunnel, dim=0).to(device=self.device)
            similarity_tunnel = similarity_tunnel.squeeze()

        # 使用掩码将需要置零的张量进行处理
        vector_matrix = vector_matrix * mask_list[0]
        data_tunnel = data_tunnel * mask_list[1]
        similarity_tunnel = similarity_tunnel * mask_list[2]
        out,weightorNone= self.decision_block(vector_matrix,data_tunnel,similarity_tunnel)
        out=self.bn1(out)
        if weightorNone !=None:
            return out,weightorNone
        return out,None
