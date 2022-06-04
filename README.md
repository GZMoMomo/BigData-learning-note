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
df_tenure_box = pd.cut(df['tenure'],bins=[-111,0,10,20,30,40,50,60,70,80,90,100,8500])
df_tenure_box.value_counts(normalize=True).sort_index()

#分箱（左包含 false）
pd.cut(range(10),bins=5,right=False)
 ```
