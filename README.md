# BigData-learning-note
大数据相关知识学习之路
## Python数据分析挖掘
## Pandas
Pandas官网 IO tools https://pandas.pydata.org/docs/user_guide/io.html  
导入科学工具包  ``` import pandas as pd ```  
### 数据读取常用方法  
```  
df = pd.read_csv('C:/Users/10624/Desktop/JAVA/python/Telco-Customer-Churn-Data.csv')  
df.head()
```  
保存csv格式数据  
```
df_csv.to_csv('**')  
#追加保存数据  
df_csv.to_csv('**',mode='a',header=None)   
```
选择列   
``` usecols=[0,2,4]```  
分段读取  
```
reader = pd.read_csv('***',chunksize=200)  
next(reader)  

## 使用iterator参数进行数据的分段读取  
reader_iterator = pd.read_csv('***',iterator=True)
chunks = []
while True:
  try:
      reader = reader_iterator.get_chunk(100)
      chunks.append(reader)
  except:
      break  
      
df_all = pd.concat(chunks)  
```
### 数据查看和预览的常用方法
设置界面显示的最大行列数据量  
```
pd.options.display.max_columns = None
```
查看数据量、数据指标、数据维度  
```
df.shape

#数据尺寸大小
df.size

#前n条数据
df.head()
#后n条数据
df.tail()
```
查看指定条件的数据

```
df[df['Partner']=='Yes'][df['gender']=='Mail']

#查看指定列
df[['Partner','gender']]

#查看指定行
df[1:30]

#查看指定列和行
df.loc[20,'gender']

#查看指定列和行
df.iat[20,4]

#查看指定列和行(可以接受索引)
df.iloc[20:30, :5]
```
查看数据的详细信息和数据指标的类型
```
df.dtypes

#统计不同类型数据指标的数量
from collections import Counter
Counter(df.dtypes.values)

#查看数据的详细信息 、指标类型统计、数据大小、内存占用信息
df.info()
```
### 数据分析和可视化
数据描述
 ```
 df.describe()
 ```
 查看每个类别数据的数量
 ```
 df['gender'].value_counts()
 
 #查看每个类别数据的占比
 df['gender'].value_counts(normalize=True)
 ```
 对数值型数据进行分箱处理
 ```
df_tenure_boxes, df_tenure_boxes_labels = pd.cut(df['tenure'],bins=[-111,0,10,20,30,40,50,60,70,80,90,100,8500],right=False,retbins=True,include_lowest=True)
df_tenure_boxes.value_counts(normalize=True).sort_index()

#分箱（左包含 false）
bins：分类依据的标准，可以是int、标量序列或间隔索引(IntervalIndex)
right：是否包含bins区间的最右边，默认为True，最右边为闭区间，False则不包含
labels：要返回的标签，和bins的区间对应
retbins：是否返回bins,当bins作为标量时使用非常有用，默认为False
precision：精度，int类型
include_lowest：第一个区间是否为左包含(左边为闭区间)，默认为False,表示不包含，True则包含
duplicates：可选，默认为{default 'raise', 'drop'}，如果 bin 边缘不是唯一的，则引发 ValueError 或删除非唯一的。
ordered：默认为True，表示标签是否有序。如果为 True，则将对生成的分类进行排序。如果为 False，则生成的分类将是无序的（必须提供标签）

pd.cut(range(10),bins=5,right=False)
 ```
```
分箱标签
df_tenure_boxes_labels
分箱标签值
df_tenure_boxes.value_counts().sort_index().values

#绘制柱状图
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.bar(range(12),df_tenure_boxes.value_counts().sort_index().values,tick_label=df_tenure_boxes.value_counts(normalize=True).sort_index().index)
plt.show()

#导入包
import seaborn as sns
import matplotlib.pyplot as plt
tips=sns.load_dataset("tips") #数据集
print(tips.shape)
tips.head(10)

#散点图
sns.scatterplot(x="total_bill",y="tip",data=tips)
plt.show()

#散点图
sns.scatterplot(df['tenure'],df['MonthlyCharges'])
plt.show()

#密度分布直方图
sns.distplot(df['tenure'])
plt.show()
```
### 数据预处理
```
#查看object类型数据的具体数据类别
#添加numpy包
import numpy as np
df_types = df.dtypes
for col in df.dtypes.index:
    if df_types[col] == object:
        print('*' * 50)
        print(col + ':')
        print(len(np.unique(df[col])))
        print(np.unique(df[col]))

#将object类型的数字转化成浮点型
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
df.info()

#查看Nan个数
df['TotalCharges'].isna().value_counts()

#查看重复个数
df['customerID'].duplicated().value_counts()
```
#### 异常值处理 （箱型图检测异常值）
1.利用四分位数来确定异常值  
四分位间距 = 上四分位数 - 下四分位数  
数据下界 = 下四分位数 - k * 四分位间距  
数据上界 = 上四分位数 - k * 四分位间距

异常值一般位于数据上界和数据下界之外，一般情况下 k 取1.5-3之间的数，k越小，正常值的范围越小，检测异常值的敏感性越高；k越大正常值的范围越大，检测异常值的敏感性越低。 
```
#利用四分位间距检测异常值(需根据业务调整)
for col in ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']:
    print('*' * 50)
    print(col)
    #求上四分位数
    q_75 = df[col].quantile(q=0.75)
    print('上四分位数：',q_75)
    #求下四分位数
    q_25 = df[col].quantile(q=0.25)
    print('下四分位数：',q_25)
    #求四分位间距
    d = q_75-q_25
    print('四分位间距：',d)
    #求数据上界和数据下界
    data_top = q_75+1.5*d
    data_bottom = q_25-1.5*d
    print('数据上界：',data_top)
    print('数据下界：',data_bottom)
    #查看异常值
    print('异常值的个数：',len(df[(df[col]>data_top)|(df[col]<data_bottom)]))
```
保留非异常数据
```
df = df[df['tenure']>=0]
print(df.shape)
```
剔除重复的字段数据
```
df = df.drop_duplicates(subset=['customerID'])
print(df.shape)
```
独热化处理
```
#独热化处理
#需要独热化处理的字段名
need_onehot_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport']

#进行独热化处理
from sklearn.preprocessing import OneHotEncoder

for col in need_onehot_cols:
    print(col)
    #创建独热化实例
    onehot=OneHotEncoder()
    #独热化训练
    onehot.fit(np.array(df[col]).reshape(-1,1))
    #获取独热化后的字段名
    new_cols = onehot.get_feature_names([col]).tolist()
    print(new_cols)
    #将独热化后的数据转化成DataFrame
    onehot_value = pd.DataFrame(onehot.transform(np.array(df[col]).reshape(-1,1)).toarray(),columns=new_cols)
    #将独热化后的数据添加到原数据里
    df[new_cols]=onehot_value
    
print(df.shape)
df.head()
```
删除经过独热化处理后的字段
```
#删除经过独热化处理后的字段
df.drop(columns=need_onehot_cols,inplace=True)
df.head(10)
```
数值化处理
```
#数值化处理
map_dict = {'Male':1,'Female':0,'Yes':1,'No':0}
df.replace(map_dict,inplace=True)
df.head(10)
```
查看空值
```
df.isna().any(axis=1).value_counts()
```
填充空值
```
df.fillna(0,inplace=True)
```
#### 数据的标准化和归一化
特点  
（1）标准化特点
对不同特征维度的伸缩变换的目的是使得不同度量之间的特征具有可比性。同时不改变原始数据的分布。好处：
1 使得不同度量之间的特征具有可比性，对目标函数的影响体现在几何分布上，而不是数值上
2 不改变原始数据的分布  

（2）归一化特点
对不同特征维度的伸缩变换的目的是使各个特征维度对目标函数的影响权重是一致的，即使得那些扁平分布的数据伸缩变换成类圆形。这也就改变了原始数据的一个分布。好处：
1 提高迭代求解的收敛速度
2 提高迭代求解的精度  

区别  
归一化：缩放仅仅跟最大、最小值的差别有关。 输出范围在0-1之间

标准化：缩放和每个点都有关系，通过方差（variance）体现出来。与归一化对比，标准化中所有数据点都有贡献（通过均值和标准差造成影响）。输出范围是负无穷到正无穷  

1.数据的标准化  
z=(x-u)/std  (u:均值）

2.数据的归一化  
minmax=(x-min)/(max-min)  

归一化处理
```
# 归一化处理
from sklearn.preprocessing import MinMaxScaler,StandardScaler
df_copy = df.copy()
cols_need_to_minmax = ['tenure','MonthlyCharges','TotalCharges']
for col in cols_need_to_minmax:
    minmax = MinMaxScaler()
    df_copy[col+'_minmax'] = minmax.fit_transform(np.array(df_copy[col]).reshape(-1,1))
    
df_copy.describe().T
```
标准化处理
```
# 标准化处理
from sklearn.preprocessing import MinMaxScaler,StandardScaler
df_copy = df.copy()
cols_need_to_standard = ['tenure','MonthlyCharges','TotalCharges']
for col in cols_need_to_standard:
    standard = StandardScaler()
    df_copy[col+'_standard'] = standard.fit_transform(np.array(df_copy[col]).reshape(-1,1))
    
df_copy.describe().T
```
# Hadoop
### MapReduce
MapReduce是一个分布式运算程序的编程框架  
- MapReduce易于编程    
- 良好的扩展性  
- 高容错性  
- 适合PB级以上海量数据的离线处理  

缺点：  
- 不擅长实时计算  
- 不擅长流式计算  
- 不擅长DAG（有向无环图）计算  
  - 多个应用程序存在依赖关系，后一个应用程序的输入为前一个的输出。在这种情况下，MapReduce并不是不能做，而是使用后，每个MapReduce作业的输出结果都会写入到磁盘，会造成大量的磁盘IO，导致性能非常的低下。 
  
MapReduce核心思想
- 分布式的运算程序往往需要分成至少2个阶段。  
- 第一个阶段的MapTask并发实例，完全并行运行，互不相干。  
- 第二个阶段的ReduceTask并发实例互不相干，但是他们的数据依赖于上一个阶段的所有MapTask并发实例的输出。  
- MapReduce编程模型只能包含一个Map阶段和一个Reduce阶段，如果用户的业务逻辑非常复杂，那就只能多个MapReduce程序，串行运行。  
MrAppMaster：负责整个程序的过程调度及状态协调。  
#### Mapper
切片与MapTask并行度决定机制  
- 数据块：Block是HDFS物理上把数据分成一块一块。数据块是HDFS存储数据单位。
- 数据切片：数据切片只是在逻辑上对输入进行分片，并不会在磁盘上将其切分成片进行存储。数据切片是MapReduce程序计算输入数据的单位，一个切片会对应启动一个MapTask。
- MapTask的并行度决定Map阶段的任务处理并发度，进而影响到整个Job的处理速度。
一个Job的Map阶段并行度由客户端在提交Job时的切片数决定。  
![image](https://user-images.githubusercontent.com/91240419/187334872-e0a2ebe5-a25c-42ef-a95c-3f083af74c56.png)
![image](https://user-images.githubusercontent.com/91240419/187334885-0e31df5f-302f-420d-a0ae-6daeab8126a0.png)

#### Shuffle
1. MapTask收集我们的map()方法输出的kv对，放到内存缓冲区中
2. 从内存缓冲区不断溢出本地磁盘文件，可能会溢出多个文件
3. 多个溢出文件会被合并成大的溢出文件
4. 在溢出过程及合并的过程中，都要调用Partitioner进行分区和针对key进行排序
5. ReduceTask根据自己的分区号，去各个MapTask机器上取相应的结果分区数据
6. ReduceTask会抓取到同一个分区的来自不同MapTask的结果文件，ReduceTask会将这些文件再进行合并（归并排序）
7. 合并成大文件后，Shuffle的过程也就结束了，后面进入ReduceTask的逻辑运算过程（从文件中取出一个一个的键值对Group，调用用户自定义的reduce()方法）
![image](https://user-images.githubusercontent.com/91240419/187335261-d8889d61-6f10-41f0-af0a-387797777742.png)

排序概述  
- MapTask处理的结果暂时放到环形缓冲区中，当环形缓冲区使用率达到80%的时候，再对缓冲区中的数据进行一次快速排序，并将这些有序数据溢写到磁盘上，当数据处理完毕后，它会对磁盘上的所有文件进行归并排序。
- ReduceTask会从每个MapTask上远程拷贝相应的数据文件，如果文件大小超过一定阈值，则溢写到磁盘上，否则存储在内存中。如果磁盘上文件数目达到一定阈值，则进行一次归并排序生成一个更大的文件，如果内存中文件大小或者数目超过一定阈值，则进行一次合并后将数据溢写到磁盘上。当所有数据拷贝完成后，ReduceTask统一对内存和磁盘上的所有数据进行一次归并排序。

### HDFS
优点
- 高容错性
- 廉价
- 大规模数据
缺点
- 不适合低延时数据
- 无法高效对大量小文件存储
- 不支持并发写入、文件随机修改（仅支持数据append追加）

架构
- NameNode：管理hdfs的名称空间、配置副本策略、管理数据库Block映射信息、处理客户端读写请求。
- DataNode：存储实际的数据块、执行数据块的读/写操作。
- Secondary NameNode：并非NameNode的热备。当NameNode挂掉的时候，它并不能马上替换NameNode并提供服务。（1.可辅助恢复NameNode 2.定期合并Fsimage和Edits，并推送给NameNode）

HDFS文件块大小
- 太小，会增加寻址时间，程序一直在找块的开始位置。
- 太大，从磁盘传输数据的时间会明显大于定位这个块的开始位置所需的时间。
- 块大小主要取决于磁盘传输速率。

HDFS写数据流程
![image](https://user-images.githubusercontent.com/91240419/187391725-3d8053b3-6ecd-4295-9414-afb7e0a97729.png)
1. 客户端通过Distributed FileSystem模块向NameNode请求上传文件，NameNode检查目标文件是否已存在，父目录是否存在。
2. NameNode返回是否可以上传。
3. 客户端请求第一个Block上传到哪几个DataNode服务器上。
4. NameNode返回DataNode节点。
5. 客户端通过FSDataOutputStream模块请求datanode节点上传数据，datanode通过pineline继续调用其他节点，将这个通信管道简历完成。
6. 各个datanode节点逐级应答客户端。
7. 客户端开始往datanode上传第一个block（先从磁盘读取数据放到一个本地内存缓存），以pakcet为单位，传递给各个节点，没传递一个packet会放入一个应答队列等待应答。
8. 当一个block传输完成之后，客户端会再次请求namenode上传第二个block的服务器

HDFS NameNode元数据存储
  - 如果元数据存储在NameNode节点的磁盘中，因为需要频繁进行随机访问，还有响应客户请求，必然效率过低。因此，元数据需要存放在内存中。但如果只存在内存中，一旦断电，元数据就会丢失。因此产生在磁盘中备份元数据的FsImage。
   - 1但当在内存中的元数据更新，同时更新FsImage，还是会效率低，如果不更新，就会发生一致性问题，一旦断电就会产生数据丢失。因此引入Edits文件（只进行追加操作，效率很高）。每当元数据有更新或添加元数据时，修改内存中的元数据并追加到Edits中。并且定期进行FsImage和Edits合并。并且引入一个新的节点SecondaryNameNode专门用于这项任务。

NameNode工作机制  
![image](https://user-images.githubusercontent.com/91240419/187452742-bcf13ca6-1ee3-4bda-a0dc-d0a71eb3a56c.png)
第一阶段：NameNode启动
1. 第一次启动NameNode格式化后，创建Fsimage和Edits文件，如果不是第一次启动，直接加载编辑日志和镜像文件到内存。
2. 客户端对元数据进行增删改的请求。
3. NameNode记录操作日志，更新滚动日志。
4. NameNode在内存中对元数据进行增删改

第二阶段：Secondary NameNode工作
1. 2NN询问NameNode是否需要CheckPoint。直接待会NameNode是否检查结果。
2. 2NN请求执行CheckPoint
3. NameNode滚动正在写的Edits日志
4. 将滚动前的编辑日志和镜像文件拷贝到2NN。
5. 2NN加载编辑日志和镜像文件到内存并合并。
6. 生成新的镜像文件fsimage.chekpoint
7. 拷贝到NameNode
8. 重命名新的镜像文件为fsimage。

