from sklearn.utils import resample  
from imblearn.over_sampling import ADASYN  
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd  
from sklearn.cluster import KMeans  
import random
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from  sklearn import metrics 

def draw_the_distribution(target, saved_path='pics/label_distribution.png'):  
    """  
    Args:  
    ------  
    target: 一个pandas的Series对象，需要统计每个取值的数量，并按比例绘制饼图。  
    saved_path: 绘制的饼图的保存路径。  
  
    Returns:  
    ------  
    None  
    """  
    value_counts = target.value_counts() # 统计每个取值的数量 
    
    total = target.size # 计算总的样本数量  
    print(value_counts,total) 
    # 绘制饼图  
    fig, ax = plt.subplots()   
    ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')   
    ax.axis('equal')  # 确保饼图是一个圆形  
  
    # 在饼图下方写上总的样本数量  
    plt.text(0, -1.2, 'total num: {}'.format(total), ha='center')  
  
    # 保存饼图  
    plt.savefig(saved_path)  
    plt.close(fig)

def split_ori_data(args,data_ori):
    """
    # 划分原始数据集，划分比例和随机种子已经写死，如果需要再使用args进行修改。
    
    Returns:
        Train_set/Test_set：已经删去了目标属性列的训练/测试集；
        Train_y/Test_y：训练/测试集中的目标属性列，原本的取值已经被替换为重新编号的index；
        Memory_set：记忆集，没有删除任何目标属性列。
    
    """
    # 划分数据 训练：测试：记忆 = 0.7:0.0.2:0.1
    Train_set, Test_set = train_test_split(data_ori, test_size=0.3, random_state=42)
    Test_set, Memory_set = train_test_split(Test_set, test_size=0.333, random_state=42)

    Train_y = Train_set[args.target_attribute]
    Train_set.drop(args.target_attribute, axis=1, inplace=True) #训练集把标签丢掉
    Test_y = Test_set[args.target_attribute]
    Test_set.drop(args.target_attribute, axis=1, inplace=True) #测试集把标签丢掉
    
    class_labels = data_ori[args.target_attribute].astype(int) #从整个数据集上获得目标属性的所有取值

    # 所有的标签可以取值的进行排序并去除重复值
    sorted_unique_values = sorted(class_labels.unique())

    # 创建一个字典，其中 键key是取值的索引，值value是目标属性 类别本身的取值
    result_dict = {index: value for index, value in enumerate(sorted_unique_values)}

    # 打印字典
    print('从整个数据集中获得的目标属性取值的字典：',result_dict)
    # 反字典得到的是 key:类别本身的取值  value:取值的index
    reverse_mapping_dict = {v: k for k, v in result_dict.items()} 
    
    Train_y1 = Train_y.replace(reverse_mapping_dict)#把原本的类别本身的取值  换成 index :Train_y1中的值变成了index
    Test_y = Test_y.replace(reverse_mapping_dict)
    print('训练集，测试集，记忆集的shape为：{}/{}/{}'.format(Train_set.shape,Test_set.shape,Memory_set.shape))
    return Train_set,Train_y1,Test_set,Test_y,Memory_set,result_dict,reverse_mapping_dict

def split_vali_data(args,Train_set,Train_y1):
    """ 
        场景应用于（已经扩充、修改好的数据集）中划分出一部分数据用于验证，验证集要和训练集保持相同的分布
        和 上面的划分数据集的不同之处在于，这个数据集并不对目标属性进行值的修改，仅仅是划分并且拆分。
    """
    Train_set, Vali_set, Train_y1,Vali_y= train_test_split(Train_set,Train_y1, test_size=0.1, random_state=42)
    print('当前数据的分布是：训练集：{}/验证集{}'.format(Train_set.shape[0],Vali_set.shape[0]))
    return Train_set, Vali_set, Train_y1,Vali_y
