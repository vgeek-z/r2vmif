from sklearn.utils import shuffle
import argparse
from sklearn.model_selection import train_test_split
import dataloader
import pickle
import numpy as np
import pandas as pd
from train import rule_embedding_train
import helper
import numpy as np
import helper
import ast  
import os
import metric
import logging
import torch
import sklearn
import sklearn.metrics
import train
from models import DataEncoder
import models
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from data.data_tools import split_ori_data,split_vali_data

# 定义命令行参数
#def parse_args():
parser = argparse.ArgumentParser(description="训练模型参数设置")

# 默认设置（文件保存路径等）
parser.add_argument("--target_attribute", type=str, default='ct_scheme', help="要进行分类推荐的属性名称")
parser.add_argument('--device', type=str, default='cuda:0',help="设置在GPU上跑")
parser.add_argument("--ckpt_ori_path", type=str, default='checkpoints/', help="保存的数据节点的路径")
parser.add_argument("--data_path", type=str, default='checkpoints/node_data/data_struct.pkl', help="保存的数据节点的路径")
parser.add_argument("--rule_ckpt_path", type=str, default=None, help="规则嵌入得到的节点向量（训练结束的）")
parser.add_argument("--ThreeTunnelsModel_ckpt_path", type=str, default='checkpoints/nets/3tunnels.pth', help="模型的权重文件")
parser.add_argument("--ThreeTunnelsModel_pics", type=str, default='logs/ThreeTunnelsModel_train_loss.png', help='模型框架训练时的损失变化')
parser.add_argument("--rule_embedding_pics", type=str, default='logs/rule_embedding_train_loss.png', help='节点嵌入时的损失变化')
parser.add_argument("--random_seed", type=int, default=42, help='设置随机种子')#42

# 三通道框架可调参数
parser.add_argument("--batch_size", type=int, default=64, help="")
parser.add_argument("--train_epoch_3tunnels", type=int, default=10, help='MIF模型训练次数')
parser.add_argument("--saved_epoch", type=int, default=15, help='相隔多少个epoch进行测试，MIF模型中')
parser.add_argument("--mask_list", type=list, default=[1,1,1], help='选择哪一个通道，分别是 通道GC，通道CC，通道NC，1为选择，0为关闭')
parser.add_argument("--learning_rate", type=float, default=0.01, help='框架的学习率，默认是0.01')#0.02

# 关于dataencoder中的参数，应该不需要再进行调整（正文有强制对齐）
parser.add_argument("--output_dim", type=int, default=64, help='初始化-output_dim') #必须和嵌入向量保持一致

# 关于规则节点嵌入中的参数
parser.add_argument("--node_embedding_dim", type=int, default=64, help='向量长度K')
parser.add_argument("--node_embedding_epoch", type=int, default=80, help='R2V训练次数')
parser.add_argument("--negative_length", type=int, default=5, help='负样本队列的长度 ')
parser.add_argument("--negative_masks", type=list, default=[1,1], help='负样本的来源：第一个元素代表 其它治疗方案，第二个代表其它AVP')
parser.add_argument("--positive_fused_weight", type=list, default=[0.33,0.33,0.33], help='属性本身+attention+卷积')#'正样本对的正样本由（1-pfw）*attention(node)+pfw*node   和 治疗方案组成')

#相似通道中节点权重
parser.add_argument("--similiar_num", type=int, default=10, help='在通道CC中选择的相似节点的个数')
parser.add_argument("--similarity_attribute_weight", type=float, default=0.5, help='近似节点融合的时候，属性和治疗方案的权重')

# 关于三通道融合的参数：
parser.add_argument("--dataencoder", type=str, default='DPCNN', help='DPCNN,MLP')
parser.add_argument("--decision_way", type=str, default='fc_adapted', help='add/fc/fc_adapted')

# 更新规则后需要更新的参数： （记得这里比实际的数值+1，因为实际的数值是从1开始编辑的）
parser.add_argument("--nums_attribute_node", type=int, default=53, help="属性节点的个数")
parser.add_argument("--nums_therapy_node", type=int, default=8, help="治疗方案节点的个数")

# 控制当下是否要重新训练 是否直接进行测试等。
parser.add_argument("--if_extend_therapy_node", type=bool, default=False, help="把所有节点嵌入都放入")
parser.add_argument("--if_ReCreate_data", type=bool, default=False, help="重新生成图")
parser.add_argument("--if_train_node_embedding", type=bool, default=False, help="节点嵌入重新训练")
parser.add_argument("--if_train_frame", type=bool, default=True, help="整个模型框架是否需要重新训练")

# 对于数据预处理的参数，包括是否进行数据筛选之类的
parser.add_argument("--if_limited_by_nodes", type=bool, default=False, help="用于检测两种可能：")

parser.add_argument("--liner_weight", type=str, default=None, help="三个通道自适应融合的权重,这里是为了方便记录，不用赋初值")
parser.add_argument("--text_info", type=str, default='提示信息记录', help="文字提示信息")
parser.add_argument("--ratio_split_create", type=list, default=[0,1,0], help='比例')
parser.add_argument("--compare_pre_train_picpath", type=str, default='baselines/compare/', help="way to balabce")
parser.add_argument("--loss_type", type=str, default='crossen', help="mse/crossen")


args = parser.parse_args()
args.output_dim = args.node_embedding_dim
if not args.rule_ckpt_path: #如果没有指定某个训练的节点嵌入的路径（也就是没指定历史权重）（就创建新的）
    path1 = os.path.join(args.ckpt_ori_path,'node_vector','')
    args.rule_ckpt_path=path1+'_D'+str(args.node_embedding_dim)+'_Epo'+str(args.node_embedding_epoch)+'_Neg'+str(args.negative_length)+'_Mask'+str(args.negative_masks[0])+str(args.negative_masks[1])+'_Pos'+str(args.positive_fused_weight[0])+str(args.positive_fused_weight[1])+str(args.positive_fused_weight[2])+'.pkl'
path1 = os.path.join(args.ckpt_ori_path,'nets','')# 改写下下游任务的保存的模型权重路径
args.ThreeTunnelsModel_ckpt_path=path1+'3tunnels'+ '_mk'+str(args.mask_list[0])+str(args.mask_list[1])+str(args.mask_list[2])+'_de'+str(args.dataencoder)+'.pth'

# 设置Log
folder_path = 'logs'
os.makedirs(folder_path, exist_ok=True)
log_file = os.path.join(folder_path, 'train_frame.log')
if os.path.exists(log_file):
    with open(log_file, 'w') as f:
        f.write('')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')#, encoding='utf-8'
logging.info('args,{}'.format(args))
print('args:',args)

# 获得原始数据
data_ori = pd.read_excel('data/data_demo.xlsx')

# 做训练前的准备
data_ori_1 = data_ori.copy() 
data_ori = data_ori_1.dropna(subset=[args.target_attribute])#删去空值
data_ori = data_ori.reset_index(drop=True) 

# 从原始数据中获得所有治疗方案的AVP组合
therapyAVP_all_list = [ ] #You can define it yourself, add it to the list manually, or add it with some custom functions

# 数据类型生成
if args.if_ReCreate_data:
    print('-- 重新生成数据：')
    if args.if_extend_therapy_node:
        data=dataloader.Create_Data(args.data_path,therapyAVP_all_list)
    else:
        data=dataloader.Create_Data(args.data_path)
    print('-- 数据生成结束!')
else:
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f) #数据类型，节点的结构'
    print('-- 读取数据结束!')

args.nums_attribute_node=len(data['attributes'].x) #更新属性节点数目
args.nums_therapy_node=len(data['therapies'].x) #更新治疗方案节点数目

# 对比学习获得嵌入表征
if args.if_train_node_embedding: #是否进行节点训练
    data_vec = rule_embedding_train(args,data,num_epochs = args.node_embedding_epoch )
else:
    with open(args.rule_ckpt_path, 'rb') as f:
        data_vec = pickle.load(f) # 通过GNN和对比学习得到的节点的向量信息。
print('-- 节点嵌入表示生成结束!')   


index_list  = list(data['attributes'].AVP_dict.keys())
index_list2 = list(data['therapies'].AVP_dict.keys())
index_list.extend(index_list2)
Ruletag_df= pd.DataFrame(columns=['tag'],index = index_list)

Train_set,Train_y1,Test_set,Test_y,Memory_set,result_dict,reverse_mapping_dict=split_ori_data(args,data_ori)
Train_set, Vali_set, Train_y1,Vali_y = split_vali_data(args,Train_set,Train_y1)
print('当前的数据集划分为：训练集：{}/测试集：{}/验证集：{}/记忆集：{}'.format(Train_set.shape[0],Test_set.shape[0],Vali_set.shape[0],Memory_set.shape[0]))

train_AVP_list =helper.create_attribute_lists(Train_set) 
    
if args.if_train_frame:
    model_trained = train.train(
        args=args,
        data_vec = data_vec,data=data,
        Train_set = Train_set,Memory_set_1 = Memory_set,train_AVP_list = train_AVP_list,Train_y1 = Train_y1,
        result_dict=result_dict,
        vali_set=Vali_set,vali_y=Vali_y,   ##验证集的修改看这里！
        reverse_mapping_dict = reverse_mapping_dict,
        vali_set_2 = Test_set, ## 同时想观察测试集
        vali_y_2 = Test_y
                                )
else:
    
    model_trained=None 

all_label,HR_Ours,NDCG_Ours,MRR_Ours = train.test(args,Test_set,Test_y,data_vec,data,Memory_set,result_dict,model_trained,reverse_mapping_dict=reverse_mapping_dict)

print(f'args.loss_type:{args.loss_type},  args.mask_list:{args.mask_list},  args.learning_rate:{args.learning_rate},  args.train_epoch_3tunnels:{args.train_epoch_3tunnels}')
print(f'**DEMO:** HR_Ours:{HR_Ours}. \n**DEMO:** MRR_Ours:{MRR_Ours}.\n**DEMO:**NDCG_Ours:{NDCG_Ours}.')
