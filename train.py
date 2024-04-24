from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import os
import sklearn
import sklearn.metrics
import torch.optim as optim
import logging
import torch.nn.functional as F
import pickle
import helper
import models
from models import GCNNet
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
import metric
def compare_with_AVP_dict(row_data, AVP_dict,continuous_attributes):
    """ 把患者数据（列表）和AVP字典进行匹配，并输出患者数据对应的节点的编号
    Args:
        row_data (_type_): 列表，[AVP1,AVP2]
        AVP_dict (_type_): 字典，里面的key一般而言对应的是 AVP ，value则对应的是这个AVP的一个编号，这个数据不来自样本集
        continuous_attributes(list)：列表，里面存放的是连续型数据的属性名称[A,B,C,D..]
        
    Returns:
        attribute_index _type_: 一个列表，对应的是row_data中存在的AVP 在AVP_dict中对应的节点的 编号。 
                                这里对离散数据和连续数据都有进行处理
    
    Needs:
        helper.find_specific_attribute
        helper.know_the_interval
    """
   
    attribute_index = []
    logging.info('***********当前的输入是：***********\n{}'.format(row_data))  
    for AVP in row_data: ## 格式： ct_scheme:5
        AV_pair_list=AVP.split(':') #把 属性，属性值 给拆开了
        
        #如果是连续型的属性
        if AV_pair_list[0] in continuous_attributes:
            AVP_sub_dict=helper.find_specific_attribute(AV_pair_list[0],AVP_dict)
            value = helper.know_the_interval(AVP,AVP_sub_dict)
            if value: #如果找到了并且不是False
                attribute_index.append(value)
                
        else:      
            for key,value in AVP_dict.items():
                if AVP in key: 
                    attribute_index.append(value)
    logging.info('可以命中的节点的index为{}'.format(attribute_index))
    return attribute_index

def rule_embedding_train(args,data,num_epochs = 10):
    """规则编码嵌入的训练过程

    Args:
        :param args (_type_): 超参数
        :param data (_type_): 异构的节点 是个数据类型（类似字典）
        :param num_epochs (int, optional): 训练次数 Defaults to 10.
        :param args.rule_ckpt_path:规则嵌入的权重
    """
    torch.manual_seed(args.random_seed)
    # 初始化模型、优化器等
    model_GCN = GCNNet(data=data,                                   # 数据
                   nums_attribute_node=args.nums_attribute_node,# 多少个属性
                   nums_therapy_node=args.nums_therapy_node,    # 治疗方案数量
                   negative_length=args.negative_length,        # 负样本队列的长度（提前设定）
                   embedding_dim=args.node_embedding_dim,       # 节点嵌入的维度
                   negative_masks=args.negative_masks           
                   ) #里面有记录LOGGING的语句
    optimizer_rule = optim.Adam(model_GCN.parameters(), lr=0.001)
    losses=[]
    for epoch in range(num_epochs):
        model_GCN.train()
        optimizer_rule.zero_grad()
        result_a,result_t,num_real_positive,loss2 = model_GCN(data,positive_confuse_way='add',fused_weight=args.positive_fused_weight) #result.shape:22 2 8

        num_samples = result_a.shape[0] 
        num_therapies = result_t.shape[0] 
        num_all = num_samples + num_therapies
        labels = torch.zeros((num_samples,result_a.shape[1]))
        labels_t = torch.zeros((num_therapies,result_t.shape[1]))
        for index,length in enumerate(num_real_positive):
            labels[index,:length]=1
        
        flattened_label = labels.view(-1).long()
        flattened_label_t = labels_t.view(-1).long()#
        flatten_all=torch.concat((flattened_label,flattened_label_t))
        flattened_embed = result_a.view(-1, result_a.size(-1))
        flattened_embed_1 = result_t.view(-1, result_t.size(-1))
        flattened_embed_all = torch.concat((flattened_embed,flattened_embed_1))
        
        result2 = model_GCN.mlp_for_gnn(flattened_embed_all) 
        
        loss1 = F.cross_entropy(result2, flatten_all)

        print(model_GCN.node_vector_dict[0].detach().numpy()[:8])
        loss_rule  = loss1+0.1*loss2
        loss_rule.backward()
        optimizer_rule.step()
        
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss1.item():.4f},{loss2.item():.4f},{loss_rule.item():.4f}')
        losses.append(loss_rule.item())
    
    helper.draw_the_loss(num_epochs,losses,args.rule_embedding_pics)    
    # 保存数据到文件   
    with open(args.rule_ckpt_path, 'wb') as f:
        pickle.dump([model_GCN.node_vector_dict,model_GCN.therapy_vector_dict], f)
    data_vector=[model_GCN.node_vector_dict,model_GCN.therapy_vector_dict]
    torch.cuda.empty_cache()
    model_GCN.zero_grad()  
    return data_vector
  
def fuse_attribute_node(data_vec, AVP_index, fuze_way='add'):
    """对一个患者有的所有节点的index，我们对应上data_vec中的tensor表征，并且用不同方法把这些表征进行融合。
        对于一个节点都没匹配上的患者，他的表征就是全0的tensor向量。
    Args:
        data_vec (dict): 训练好的包含AVP节点嵌入表示的字典，字典的key是AVP的编号，字典的Value是对应AVP节点的嵌入表示
                        例如: {0: tensor([0.0849, -1.8363, 0.2706, ...], requires_grad=True)}
        AVP_index (list): 包含整数编号的列表，代表要融合的AVP节点的索引，例如 [1, 2, 5, ...]，也就是当前患者的所有的节点
        fuze_way (str): 融合方式，可以选择在['add', 'mean']中。

    Returns:
        fused_vector: 一个融合后的tensor向量
    """
    if AVP_index ==[]: #如果对应不上任何一个节点，就置为0。
        _, first_value = next(iter(data_vec.items()))
        dim = len(first_value)
        return torch.zeros(dim)
    
    if fuze_way == 'add':
        fused_vector = torch.sum(torch.stack([data_vec[i] for i in AVP_index]), dim=0)
    elif fuze_way == 'mean':
        fused_vector = torch.mean(torch.stack([data_vec[i] for i in AVP_index]), dim=0)
    else:
        raise ValueError("Unsupported 'fuze_way'. Choose from ['add', 'conv', 'nn']")

    return fused_vector

def Find_similiar_samples(sample,rule_dataset_df,similiar_num=5,simliar_approach=True):
    """
        用在第三个通道中，寻找和输入最相关的N个例子。
        
        Args
        -------
            sample: 输入需要相似样本的目标样本，是一个二维的列表，里面只有该样本的值
            
            dataset:寻找相关的数据集，可以使用df.values
            
            similiar_num:相似个数，找相似的n个样本
            
            similiar_approach: 当是True的时候，就说明可以选择样本和数据集中最相似的n个，当是false的时候，就选择最相似的n个向量
        
        Returns
        -------
            similary_top_n_samples:返回样本的值，三维列表，（第x个sample ，第j相似的样本，样本的值）
            
            similarity_top_n:概率
            
            top_indices:在样本集中的index，也是一个二维列表，行代表第i个样本，列代表第j相似的样本的index
        
    """
    
    if simliar_approach:
        rule_dataset=rule_dataset_df.values
        similarity_=sklearn.metrics.pairwise.cosine_similarity(sample,rule_dataset)
        top_indices = np.argsort(-similarity_, axis=1)[:, :similiar_num]
        top_indices_index = rule_dataset_df.index[top_indices[0]]
        sample_size = len(sample)
        similary_top_n_samples=[[] for _ in range(sample_size)] 
        similarity_top_n = [[] for _ in range(sample_size)]
        for i in range(sample_size):
            for j in range(similiar_num):
                index=top_indices[i][j] 
                similary_top_n_samples[i].append(rule_dataset[index])
                similarity_top_n[i].append(similarity_[i,index])
        similary_top_n_samples = np.array(similary_top_n_samples)
        similarity_top_n = np.array(similarity_top_n)
        for i in range(len(similarity_top_n)):
            similarity_top_n[i] = helper.softmax(similarity_top_n[i])
        return top_indices_index , similary_top_n_samples,similarity_top_n
    else:
        data_vec=rule_dataset_df 
        m1 = sample.detach().numpy()
        values = [v.detach().numpy() for v in data_vec[1].values()]  
        values_array = np.array(values)  
        similarity_=sklearn.metrics.pairwise.cosine_similarity(m1,values_array)
        top_indices = np.argsort(-similarity_, axis=1)[:, :similiar_num]
        return top_indices 
  

def similary_vector_fused(vector_list,fuse_way='add',similiar_num=None,weighted_values=None):
    """这个函数是用来融合相似节点的 也就是通道三中的相似节点融合！

    Args:
        vector_list (_type_): 这是一个列表，里面保存的是（与样本）相似N个tensor向量_
        fuse_way (str, optional): 选择相似节点的融合方式，可以选择['add','WeightedAdd','conv']. Defaults to 'add'.
        
    Returns:
        返回一个融合后的向量，形状应该是[1,相似向量的维度]
    """
    if not vector_list:
        return torch.zeros(1, 0)

    if fuse_way == 'add':
        fused_vector = torch.stack(vector_list).sum(dim=0)
        fused_vector=fused_vector/similiar_num
    elif fuse_way =='WeightedAdd':
        for i in range(len(weighted_values)): #前N个
            vector_list[i] = weighted_values[i]*vector_list[i]
        fused_vector = torch.stack(vector_list).sum(dim=0)
    else:
        raise ValueError("Invalid fuse_way. Supported options are 'add' and 'conv'.")
    return fused_vector.unsqueeze(0)

def train(args,data_vec,data,Train_set,Memory_set_1,train_AVP_list,Train_y1,result_dict,vali_set,vali_y,reverse_mapping_dict,vali_set_2=None,vali_y_2=None): 
    torch.manual_seed(args.random_seed)

    lables_num = len(result_dict)    
    attribute_list = list(Train_set.columns) 
   
    model_3tunnels = models.Three_Tunnels(
        data_vec=data_vec,
        data=data,
        Memory_set=Memory_set_1,
        attribute_list=attribute_list,
        target_attribute=args.target_attribute,
        DataEncoder_Output_Dim=args.output_dim,
        fused_hidden=args.node_embedding_dim,
        fused_output=lables_num,
        dataencoder = args.dataencoder,
        decision_way=args.decision_way)#add,fc
    
    model_3tunnels.to(args.device)
    optimizer =optim.SGD(model_3tunnels.parameters(), lr=args.learning_rate)   
    loss_3_tunnels = 0
    losses=[]
    SEED = 42
    torch.manual_seed(SEED)
    batch_size = args.batch_size
    batches = [(train_AVP_list[i:i+batch_size], Train_set[i:i+batch_size], Train_y1[i:i+batch_size]) 
               for i in range(0, len(train_AVP_list), batch_size)]
    for epoch in range(args.train_epoch_3tunnels):
        model_3tunnels.train()
        total_loss = 0.0
        for batch in batches:
            avp_batch, train_set_batch, train_y1_batch = batch
            out,weightORNone = model_3tunnels(avp_batch, train_set_batch, args.similiar_num, mask_list=args.mask_list,
                        weight1=args.similarity_attribute_weight)
            if weightORNone!=None:
                args.liner_weight = weightORNone.cpu().detach().numpy()
            else:
                args.liner_weight = weightORNone
            labels = [int(i) for i in train_y1_batch.values]        
            y_index_in_data = [] 
            for i in labels:
                real_value = result_dict[i]
                for  key,value in data['therapies'].index_dict.items():
                    if value.split(':')[-1] == str(real_value) and value.split(':')[0] ==args.target_attribute:
                        y_index_in_data.append(key)
                        break
            if args.loss_type=='mse':
                label_vect = torch.stack([data_vec[1][i] for i in y_index_in_data]).to(args.device)
                diff = F.pairwise_distance(out,label_vect,p=1)
                loss_3_tunnels = model_3tunnels.loss_MSE(diff,torch.zeros_like(diff))
            elif args.loss_type=='crossen':
                labels_tensor = torch.tensor(train_y1_batch.values).long().to(args.device)#方便进行交叉熵loss
                result = model_3tunnels.fc_hid2classnum(out)#result.shape
                loss_3_tunnels = model_3tunnels.loss_CrossEntropy(result,labels_tensor)
            elif args.loss_type=='v_?':
                labels_tensor = torch.tensor(train_y1_batch.values).long().to(args.device)#方便进行交叉熵loss
                result = model_3tunnels.fc_hid2classnum(out)#result.shape
                loss_3_tunnels = model_3tunnels.loss_CrossEntropy(result,labels_tensor)
            total_loss += loss_3_tunnels.item()         
            optimizer.zero_grad()
            loss_3_tunnels.backward()
            optimizer.step()
            
        for param_name, param in model_3tunnels.named_parameters():
            if torch.isnan(param).any():
                print(f"Training:epoch{epoch}:diff{diff}\nloss_3_tunnels{loss_3_tunnels}\n:NaN found in parameter {param_name} during initialization")
            
        if epoch % args.saved_epoch==0:
            print('***验证集1***')
            vali_frame_work(args,model_3tunnels,vali_set,vali_y,data_vec,data,result_dict=result_dict,reverse_mapping_dict=reverse_mapping_dict,info='vali_set_epoch'+str(epoch))
            print('***验证集2***')
            vali_frame_work(args,model_3tunnels,vali_set_2,vali_y_2,data_vec,data,result_dict=result_dict,reverse_mapping_dict=reverse_mapping_dict,info='test_set_epoch'+str(epoch))
        losses.append(total_loss / len(batches))
        print(f'Epoch [{epoch + 1}/{args.train_epoch_3tunnels}] - Loss: {losses[-1]}')
        
    helper.draw_the_loss(args.train_epoch_3tunnels,losses,args.ThreeTunnelsModel_pics) 
    print('已绘制！')
    torch.save(model_3tunnels.state_dict(),args.ThreeTunnelsModel_ckpt_path)  
    print('已保存')
    return model_3tunnels
    
def test(args,Test_set,Test_y,data_vec,data,Memory_set_1,result_dict,model_pre=None,reverse_mapping_dict=None):
    lables_num = len(result_dict)    # 分类的次数
    attribute_list = list(Test_set.columns)
    model = models.Three_Tunnels(
        data_vec=data_vec,
        data=data,
        Memory_set=Memory_set_1,
        attribute_list=attribute_list,
        target_attribute=args.target_attribute,
        DataEncoder_Output_Dim=args.output_dim,
        fused_hidden=args.node_embedding_dim,fused_output=lables_num,
        dataencoder = args.dataencoder,
        decision_way=args.decision_way)
    model.to(args.device)
    
    if args.if_train_frame: #如果要重新训练 就不加载模型权重 
        model=model_pre
    else: #如果没有重新进行训练 才需要试用之前保存下的模型
        model.load_state_dict(torch.load(args.ThreeTunnelsModel_ckpt_path))  #加载之前保存的模型
    model.eval()
    print('**测试**')
    test_AVP_list =helper.create_attribute_lists(Test_set)
    out,weightORNone = model(test_AVP_list,Test_set, args.similiar_num, mask_list=args.mask_list,
                        weight1=args.similarity_attribute_weight)
    if weightORNone!=None:
        args.liner_weight = weightORNone.cpu().detach().numpy()
    else:
        args.liner_weight = weightORNone
    labels = [int(i) for i in Test_y.values]
    y_index_in_data = []
    for i in labels:
        real_value = result_dict[i]
        for  key,value in data['therapies'].index_dict.items():
            if value.split(':')[-1] == str(real_value) and value.split(':')[0] ==args.target_attribute:
                y_index_in_data.append(key)
                break  
    if args.loss_type=='mse':
        label_vect = torch.stack([data_vec[1][i] for i in y_index_in_data]).to(args.device)
        diff = label_vect-out
        loss = model.loss_MSE(diff,torch.zeros_like(diff))
    if args.loss_type=='crossen':
        result = model.fc_hid2classnum(out)
        labels_tensor = torch.tensor(Test_y.values).long().to(args.device)#交叉熵要用的tensor类型
        loss = model.loss_CrossEntropy(result,labels_tensor)
    if args.loss_type=='mse':
        therapy_vec_stack = torch.stack(list(data_vec[1].values())).cpu().detach()
        predicted_classes = helper.predict_classes(out.cpu(),therapy_vec_stack,k=6,dis_format=1)
        predicted_classes = helper.vali_predict_label(predicted_classes,data['therapies'].index_dict,Test_y.name)    
        all_label = []
        for i in predicted_classes: 
            sub_list = []
            for j in i: 
                real_label = data['therapies'].index_dict[int(j)].split(':')[-1] 
                real_label = reverse_mapping_dict[int(real_label)] 
                sub_list.append(real_label)
            all_label.append(sub_list)
    if args.loss_type=='crossen':  
        top_n, predicted_classes = torch.topk(result, k=5, dim=1)  
        predicted_classes=predicted_classes.cpu().detach().int().tolist()
        all_label=predicted_classes
    

    metric_dict={}    
    HR_values,NDCG_values,MRR_values=[],[],[]
    K=5
    for k in range(1,K+1):
        HR_v = metric.HR_at_K(Test_y.values,all_label,k=k)
        NDCG_v = metric.Normalized_Discounted_Cumulative_Gain(Test_y.values,all_label,k=k)
        MRR_v = metric.Mean_Reciprocal_Rank(Test_y.values,all_label,k=k)
        HR_values.append(HR_v)
        NDCG_values.append(NDCG_v)
        MRR_values.append(MRR_v)
        print('当前HR@{}是{}。'.format(k,HR_v))
        print('当前NDCG@{}是{}。'.format(k,NDCG_v))
        print('当前MRR@{}是{}。\n'.format(k,MRR_v))
    metric_dict['HR']=HR_values
    metric_dict['MRR_values']=MRR_values
    metric_dict['NDCG_values']=NDCG_values
    return all_label,HR_values,NDCG_values,MRR_values

def vali_frame_work(args,model,Test_set,Test_y,data_vec,data,result_dict=None,reverse_mapping_dict=None,info=None):
    model.eval()
    print('**验证精确度**')
    test_AVP_list =helper.create_attribute_lists(Test_set)
    for param_name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in parameter {param_name}.！！")
    out,weightORNone = model(test_AVP_list,Test_set, args.similiar_num, mask_list=args.mask_list,
                        weight1=args.similarity_attribute_weight)
    if weightORNone!=None:
        args.liner_weight = weightORNone.cpu().detach().numpy()
    else:
        args.liner_weight = weightORNone 
    labels = [int(i) for i in Test_y.values]  # index
    y_index_in_data = [] 
    for i in labels:
        real_value = result_dict[i]#这个值是属性的取值
        for  key,value in data['therapies'].index_dict.items():
            if value.split(':')[-1] == str(real_value) and value.split(':')[0] ==args.target_attribute:
                y_index_in_data.append(key)
                break
            
    if args.loss_type=='mse':
        label_vect = torch.stack([data_vec[1][i] for i in y_index_in_data]).to(args.device)
        diff = label_vect-out
        loss = model.loss_MSE(diff,torch.zeros_like(diff))
    if args.loss_type=='crossen':
        result = model.fc_hid2classnum(out)
        labels_tensor = torch.tensor(Test_y.values).long().to(args.device)
        loss = model.loss_CrossEntropy(result,labels_tensor)
    print(f'##当前的{args.loss_type}损失是{loss}')  
    if args.loss_type=='mse':
        s_temp = torch.stack(list(data_vec[1].values())).cpu().detach()
        predicted_classes = helper.predict_classes(out.cpu(),s_temp,k=6,dis_format=1)
        predicted_classes = helper.vali_predict_label(predicted_classes,data['therapies'].index_dict,Test_y.name)    
        all_label = []
        for i in predicted_classes: 
            sub_list = []
            for j in i: 
                real_label = data['therapies'].index_dict[int(j)].split(':')[-1]
                
                real_label = reverse_mapping_dict[int(real_label)] 
                sub_list.append(real_label)
            all_label.append(sub_list)
    if args.loss_type=='crossen':  
        top_n, predicted_classes = torch.topk(result, k=5, dim=1)  
        predicted_classes=predicted_classes.cpu().detach().int().tolist()  
        all_label=predicted_classes
    metric_dict={}    
    
    HR_values,NDCG_values,MRR_values=[],[],[]
    K=5
    for k in range(1,K+1):
        HR_v = metric.HR_at_K(Test_y.values,all_label,k=k)
        NDCG_v = metric.Normalized_Discounted_Cumulative_Gain(Test_y.values,all_label,k=k)
        MRR_v = metric.Mean_Reciprocal_Rank(Test_y.values,all_label,k=k)
        HR_values.append(HR_v)

        NDCG_values.append(NDCG_v)
        MRR_values.append(MRR_v)
        print('当前HR@{}是{}。'.format(k,HR_v))
        print('当前NDCG@{}是{}。'.format(k,NDCG_v))
        print('当前MRR@{}是{}。\n'.format(k,MRR_v))
    metric_dict['HR']=HR_values
    metric_dict['MRR_values']=MRR_values
    metric_dict['NDCG_values']=NDCG_values
    return all_label,HR_values,NDCG_values,MRR_values
    
