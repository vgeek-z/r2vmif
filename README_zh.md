# 1、项目环境
输入以下代码创建虚拟环境 R2V_MIF

    conda env create -f environment.yaml

# 2、项目目录结构
```
.
├── checkpoints                # 模型权重文件
│   ├── nets                   
│   ├── node_data
│   └── node_vector
├── data
│   ├── data_demo.xlsx          *注1 
│   └── data_tools.py
├── logs
├── args_demo.txt              # 参数参考 *注2
├── dataloader.py
├── environment.yaml
├── helper.py                  # 工具类函数
├── main.py                    # 项目代码入口
├── metric.py                  # 评价指标
├── models.py                  # R2V-MIF 神经网络模型结构
├── README_zh.md
├── README.md
└── train.py                   # 模型训练、预测
```
*1：只展示部分数据作为示例, 数据可以从BCDB官网下载：http://bcdb.mdt.team:8080
*2：参数作为参考（在论文中提及的数据结构下）。

# 3、进行训练和预测
    python main.py

# 4、预期运行结果
## 4.1 进度提示信息
包括生成/读取规则图提示，R2V规则节点嵌入生成提示信息，数据分布，训练进度等。
## 4.2 模型保存
包括3个权重文件，分别是规则图数据文件，规则节点嵌入权重文件，MIF权重文件，分别保存在以下目录：
 checkpoints/node_data, checkpoints/node_vector, checkpoints/nets。
## 4.3 预测结果
使用3个评价指标，HR@K，MRR@K，NDCG@K，并在运行时产生提示信息。

# 5、其它信息
1、考虑到隐私保护等原因，医疗数据不进行上传，可向BCDB官网（http://bcdb.mdt.team:8080）申请下载。



