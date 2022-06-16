# BigData-learning-note
大数据相关知识学习之路
# Python数据分析挖掘
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
