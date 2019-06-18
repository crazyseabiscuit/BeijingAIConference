
# coding: utf-8

# # 数据预处理

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pylab

# ## 读取数据
# - data_all是Lending Club 2016年的部分借贷数据

raw_df = pd.read_csv('../data/data_all.csv')
print '数据规模:',raw_df.shape


# ## 删除如下的特征
# - 非数值型特征
# - 对识别欺诈无意义的特征：id, member_id

reamin_features =[feat for feat in raw_df.select_dtypes(include=['float64', 'int64']).keys()                   if feat not in ['id', 'loan_status', 'member_id','issue_d']]
feature_df = raw_df[reamin_features]
df = raw_df.copy()
df = df[reamin_features+['loan_status','issue_d']]


# ## 将月份转化为数值


def map_month(x):
    """ Map the month strings to integers.
    """
    if x!=x:
        return 0
    if "Jan" in x:
        return 1
    if "Apr" in x:
        return 4
    if 'Aug' in x:
        return 8
    if 'Dec' in x:
        return 12
    if 'Feb' in x:
        return 2
    if 'Jul' in x:
        return 7
    if 'Jun' in x:
        return 6
    if 'Mar' in x:
        return 3
    if 'May' in x:
         return 5
    if 'Nov' in x:
        return 11
    if 'Oct' in x:
        return 10
    if 'Sep' in x:
        return 9


df.issue_d = map(map_month,df.issue_d)
month_max = max(df.issue_d)
print '一共'+str(month_max)+'个月的数据'


# ## 统计各特征的缺失比例
# - 这里不包括计算KS、IV、PSI的标签数据: loan_status, issue_d


def get_nan_cnt(feature_df):
    """feature_df is a data frame.
       return the missing value counts of every feature.
    """
    nan_cnt = []
    nan_cnt =  (feature_df!=feature_df).sum(axis=0)
    return nan_cnt


nan_cnt = get_nan_cnt(feature_df)
total = raw_df.shape[0]
nan_cnt = nan_cnt *1.0 / total
nan_df = pd.DataFrame(nan_cnt,columns=['nan_ratio'])
nan_df.index.name = 'feature'
print '缺失比例最高的一些特征：'
print nan_df.sort_values(by='nan_ratio',ascending=False).head(20)


# ## 输出缺失比例和处理后的数据

df.to_csv('../data/data_clean.csv',index=False)
nan_df.to_csv('../output/Completeness.csv')
