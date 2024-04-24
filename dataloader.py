
# 引入相关的包
import pandas as pd
import numpy as np
import logging

# 制作图
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from torch_geometric.data import HeteroData
import pickle
import os

def create_the_relationship(therapy_node,r1_therapy,r1_attribute,O_list,D_list,attribute_AVP_dict,therapy_AVP_dict):    
    if therapy_node in r1_therapy:
        for attribute_node in r1_attribute: 
            O_list.append(attribute_AVP_dict[attribute_node])
            D_list.append(therapy_AVP_dict[therapy_node])
    return O_list,D_list

def add_the_rule_tag_for_node(attribute_node,r1_attribute,rule_tag,attribute_rule,attribute_AVP_dict):
    if attribute_node in r1_attribute:
            attribute_rule[attribute_AVP_dict[attribute_node]].append(rule_tag)
    return attribute_rule

def Create_Data(data_path,exter_therapies=[]):
    folder_path = 'logs'
    os.makedirs(folder_path, exist_ok=True)

    # 设置日志文件路径
    log_file = os.path.join(folder_path, 'data_create.log')
    if os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('ssss')
    # 配置日志记录器
    logger = logging.getLogger('111')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')#, encoding='utf-8'

    # 4-1
    r1_attribute=['pathological_type:2']
    r1_therapy=['is_ct:0']
    
    
    # 3-6#
    r10_attribute=['pathological_type:0','pathological_type:1','pathological_type:4','pathological_type:5',
                'pathological_type:6','pathological_type:8',
                'er_value>=1','1<=pr_value<20','pr_value>=20',
                'cerbb_2:0','cerbb_2:1',
                'her2_fish:0',
                'tumor_phase:2','tumor_phase:3','tumor_phase:8','tumor_phase:9',
                'node_phase:0',   
                '18<=rs_21_gene<=24', '24<rs_21_gene<=30','rs_21_gene>30',
                'histological_grade:5', 
                'tumor_phase:2', 
                'tumor_phase:3', 
                'age<50',
                'ki67_value>=20',
                '1<=pr_value<20',
                'pr_value<1'
                ]
    r10_therapy=['ct_scheme:8'] 
    
    
    continuous_attributes=['er_value','pr_value','her2_fish','age','ki67_value','rs_21_gene']

    # TODO：新增规则的时候需要：把出现的节点加入到对应的列表中
    attribute_list = []#寻找并存放所有属性AVPs
    for list_attribute in [r1_attribute,r10_attribute]:# TODO:此处加入 r?_attribute
        attribute_list.extend(list_attribute)
    logger.info('所有的属性节点为：\n{}'.format(attribute_list))
    
    therapy_list = []#寻找并存放所有治疗方案AVPs
    for list_therapy in [r1_therapy,r10_therapy]:# TODO:此处加入 r?_therapy
        therapy_list.extend(list_therapy)

    if exter_therapies == []:
        pass#没有进行治疗方案的拓展，所有的治疗方案节点都来自于规则。
    else:
        therapy_list.extend(exter_therapies)
    
    logger.info('所有的治疗方案节点为：\n{}'.format(therapy_list))

    # 列表
    attribute_list=list(sorted(set(attribute_list)))
    therapy_list=list(sorted(set(therapy_list)))
    
    # 字典形式表示节点
    attribute_AVP_dict = {}
    attribute_index_dict = {}
    for i in range(len(attribute_list)):
        attribute_AVP_dict[attribute_list[i]]=i  # key:value 也就是 AVP:index
        attribute_index_dict[i]=attribute_list[i] # key:value 也就是 index:AVP

    therapy_AVP_dict = {} 
    therapy_index_dict = {}
    for i in range(len(therapy_list)):
        therapy_AVP_dict[therapy_list[i]]=i  # key:value 也就是 AVP:index
        therapy_index_dict[i]=therapy_list[i] # key:value 也就是 index:AVP
    logger.info('治疗方案节点的字典（正/倒）：\n{}\n{}'.format(therapy_AVP_dict,therapy_index_dict))

        
    O_list=[]
    D_list=[]

    for therapy_node in therapy_list:  # 如果治疗方案节点在规则列表中
        O_list,D_list=create_the_relationship(therapy_node,r1_therapy,r1_attribute,O_list,D_list,attribute_AVP_dict,therapy_AVP_dict)
        O_list,D_list=create_the_relationship(therapy_node,r10_therapy,r10_attribute,O_list,D_list,attribute_AVP_dict,therapy_AVP_dict)
                      
    logger.info('当前的边的结构是：\n{}\n{}'.format(O_list,D_list))

    attribute_rule={} 
    therapy_rule = {} 
    for attribute_node in attribute_AVP_dict.keys():
        attribute_rule[attribute_AVP_dict[attribute_node]]=[]
        attribute_rule=add_the_rule_tag_for_node(attribute_node,r1_attribute,1,attribute_rule,attribute_AVP_dict)

        attribute_rule=add_the_rule_tag_for_node(attribute_node,r10_attribute,2,attribute_rule,attribute_AVP_dict)
    for therapy_node in therapy_AVP_dict.keys():
        therapy_rule[therapy_AVP_dict[therapy_node]]=[]
        therapy_rule=add_the_rule_tag_for_node(therapy_node,r1_therapy,1,therapy_rule,therapy_AVP_dict)

        therapy_rule=add_the_rule_tag_for_node(therapy_node,r10_therapy,2,therapy_rule,therapy_AVP_dict)

    # 保存为data类。
    edge_index= torch.tensor([O_list,D_list]) # 边信息
    data = HeteroData()
    data['attributes'].x=torch.tensor(list(attribute_AVP_dict.values()))
    data['attributes'].AVP_dict = attribute_AVP_dict
    data['attributes'].index_dict=attribute_index_dict
    data['attributes'].rule_tag=attribute_rule
    data['attributes'].continuous_attributes= continuous_attributes #连续属性
    
    data['therapies'].x=torch.tensor(list(therapy_AVP_dict.values()))
    data['therapies'].AVP_dict = therapy_AVP_dict
    data['therapies'].index_dict = therapy_index_dict
    data['therapies'].rule_tag=therapy_rule
    data['attribute_to_therapies'].edge_index = edge_index 
    #增加新规则的时候记得增加这里 。
    data['rules'] = [1,2]
    logger.info('现在的数据为：{}'.format(data))    
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    return data
#Create_Data()