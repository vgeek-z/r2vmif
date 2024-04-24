# 这个文件存储一些工具类函数
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def int_to_binary_vector(num, num_bits=8):
    binary_str = format(num, f'0{num_bits}b')  # 转换为指定位数的二进制字符串
    binary_vector = [int(bit) for bit in binary_str]  # 转换为整数列表
    return ''.join(map(str, binary_vector))

def return_rule(data):
    rule = []
    attribute_node = data['attributes'].x 
    for rule_tag in data['rules']: # 对于每一个规则编码
        list_sub=[]
        for node in attribute_node: #对于每一个AVP节点
            if rule_tag in data['attributes'].rule_tag[int(node)]:
                list_sub.append(int(node))
        rule.append(list_sub)
    print(rule)
    return rule

def create_attribute_lists(dataframe):
    """把dataframe的表格转换为list,也就是把每行变成一个列表，每行变成AVP的形式，整张表格就是[[行1的AVPs],[行2的AVPs]..]

    Args:
        dataframe (_type_): 数据表

    Returns:
        attribute_lists (_type_): 一个列表，列表中每个子列表存储的是一个样本所有的AVP  e.g. 'is_ct:1', 'ct_scheme:5',
    """
    attribute_lists = []
    for index, row in dataframe.iterrows():
        attribute_list = []
        for column in dataframe.columns:
            if not pd.isna(row[column]):
                value = row[column]
                if isinstance(value, float) and value.is_integer():
                    value = int(value)  
                attribute = f"{column}:{value}" 
                attribute_list.append(attribute)
        attribute_lists.append(attribute_list)
    return attribute_lists

def find_specific_attribute(attribute, AVP_dict):
    """
    一个小函数，
        输入是字典和一个属性名
        输出是字典中所有包含该属性名的 键值对
        用字典的形式表现输出
        
    Args:
        attribute (str): str属性名
        AVP_dict (dict): AVP的键值对，一般是  attribute:value
    """
    result_dict = {}
    
    for key, value in AVP_dict.items():
        if attribute in key:
            result_dict[key] = value
    return result_dict


def know_the_interval(patient_info, AVP_subdict):
    """
    一个小函数
    输入是 字典 和 字符串
    目的是确定该字符串是否匹配字典中的key
    输出是 匹配的结果 也就是字典中的value
    
    Args:
    patient_info (str): 输入的字符串，表示病人信息，格式如 'a:0.3'
    AVP_subdict (dict): 包含匹配规则和对应值的字典，例如 {'0.1<a<0.5': 0, '0.5<=a<1': 1, 'a>=1': 2}
    
    Returns:
    int or False: 如果找到匹配的规则，返回匹配的结果值（int），如果没有匹配的规则，返回 False
    
    Example:
    >>> know_the_interval({'0.1<a<0.5': 0, '0.5<=a<1': 1, 'a>=1': 2}, 'a:0.3')
    0
    
    >>> know_the_interval({'0.1<a<0.5': 0, '0.5<=a<1': 1, 'a>=1': 2}, 'a:-1')
    False
    """
    
    # 将输入的字符串按冒号分割成属性名和属性值
    attribute, value = patient_info.split(':')
    
    for rule in AVP_subdict.keys(): # rule '0.1<a<0.5'
        # 使用eval函数解析规则，注意这里需要小心安全问题
        try:
            if eval(rule.replace(attribute, value)):
                return AVP_subdict[rule]
        except:
            pass
    
    # 如果没有匹配的规则，则返回False
    return False

def softmax(x):
    """计算softmax

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    row_max = np.max(x)
    x = x - row_max
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp)
    s = x_exp / x_sum
    return s

def draw_the_loss(num_epochs,losses,loss_pics_path):
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # 保存损失曲线图像到文件夹
    plt.savefig(loss_pics_path)
    plt.close()
    
def vali_predict_label(predict_label,therapydict_index2value,target_attribute):
    """
        Args
        -------
            predict_label: therapy_index,just like in data['therapies']  e.g..[[1,2,3,4,5],[1,3,4,5,2],[2,3,4,5,1]]
            
            therapydict_index2value: 一个字典，要求为 index:AVP     e.g..0:'is_ct:0'
            
            target_attribute: default: 要进行分类的治疗方案的类别, e.g..ct-chema
        
        Returns
        -------
            target_labels : 排除了非目标属性后的真实的值
    """  
    target_labels = []
    for j in predict_label:#对于每一个样本
        sub_target_labels = []
        for i in j:#具体的data中的index
            therapy_AVP = therapydict_index2value[int(i)]
            therapy_attribute = therapy_AVP.split(':')[0]
            if therapy_attribute != target_attribute:
                pass
            else:
                sub_target_labels.append(int(i))
        target_labels.append(sub_target_labels) 
    return target_labels   
    
import matplotlib.pyplot as plt  
def draw_the_metric(data_list, labels, filename='a.png',xlabel='Top K',ylabel='HR@K'):    
    """    
    Plot data in the same figure.    
        
    Parameters:    
    data_list (list of floats): The data to plot. Each sublist is a set of data.  
    labels (list of string): 每个数据组的标签  
      
    Returns:    
    None    
    """    
    try:    
        plt.close(plt.gcf())    
    except:    
        pass    
    colors = ['red', 'green',  'orange','blue','pink','black']
    markers = ['s','^','o','x']
    for i, data in enumerate(data_list):    
        plt.plot(data, label=f'{labels[i]}',color=colors[i % len(colors)], marker=markers[i % len(markers)])    
        # 对于每个数据点添加注释  
        for j, point in enumerate(data):  
            plt.annotate(f'{point:.2f}', xy=(j, point), xytext=(5,5), textcoords='offset points')  
    #plt.ylim(0,1)  
    plt.xticks(range(0,len(data_list[0])),range(1,len(data_list[0])+1))           
    plt.legend()    
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)  
    plt.savefig(filename,dpi=500)
    
def get_therapyAVP_all_list(data_ori,target_attribute):
    # 获得所有的治疗方案
    target_column = pd.DataFrame(data_ori[target_attribute])  
    therapyAVP_all = create_attribute_lists(target_column)
    therapyAVP_all_list = []
    for i in therapyAVP_all:
        therapyAVP_all_list.append(i[0])
    therapyAVP_all_list=list(set(therapyAVP_all_list))
    
def convert2csv(df,output_filename,target_attribute):
    """
    把datafframe转换为csv。
    第一列是label，第二列是患者的数据，以字典的形式保存。
    """
    df_new = pd.DataFrame(columns=['1','2'])
    for index, row in df.iterrows():
        if np.isnan(row[target_attribute]):
            continue

        sample_dict = {}
        for column in df.columns:
            if column != target_attribute and not pd.isnull(row[column]):
                sample_dict[column] = row[column]
        df_new.loc[len(df_new)] = [row[target_attribute], sample_dict]
    df_new.to_csv(output_filename,columns=None,index=False) 
import pandas as pd
import json
import numpy as np

def convert_df_to_json(df, output_filename,target_attribute):
    # Initialize an empty dictionary for each class
    data_dict = {str(int(i)): [] for i in df[target_attribute].unique() if not np.isnan(i)}

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        # Skip rows where the label is NaN
        if np.isnan(row[target_attribute]):
            continue

        # Create a dictionary for this sample
        sample_dict = {}
        for column in df.columns:
            # Skip NaN values and the label column
            if column != target_attribute and not pd.isnull(row[column]):
                # Add this attribute to the sample dictionary
                sample_dict[column] = row[column]

        # Add the sample dictionary to the appropriate class list
        data_dict[str(int(row[target_attribute]))].append({"text": sample_dict})#, "label": int(row[args.target_attribute])

    # Save the data dictionary as a JSON file
    with open(output_filename, 'w') as f:
        json.dump(data_dict, f)
        
import torch
import torch.nn.functional as F
      
def predict_classes(model_vec,therapy_vec,k=5,dis_format=1):
    """
    model_vec: 模型产出的患者的vect表示
    therapy_vec： 所有治疗方案的vec表示
    dis_format: 距离范式，当为2的时候，计算的是欧式距离
    """
    predict_k=[]
    for i in range(len(model_vec)): #对于每一个患者i产生的模型的预测向量。
        patient_i = model_vec[i]
        dis_i = F.pairwise_distance(patient_i,therapy_vec,p=dis_format)
        values,indices = torch.topk(dis_i,k=len(dis_i),largest = False) #这里是从大到小排序
        predict_k.append([int(i) for i in indices][:k])
    return predict_k

def sigmoid(x):
    return 1 / (1 + np.exp(-x))