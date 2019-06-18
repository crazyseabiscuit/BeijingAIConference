
# coding: utf-8

"""计算不同特征的KS和IV值，并进行比较
"""

from __future__ import division
import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pylab

# 1. 计算单个变量的KS和IV值
# 1.1 读数据
# 
# KS_IV_example.csv是课上计算KS和IV案例的数据，每一行都表示一个账户
# 
# 数据包括2个字段：category， score
# category = 0/1 表示账户的好/坏
# score 表示账户的信用评分

print '====================================='
print '计算单个变量的KS和IV值'
print '====================================='

df = pd.read_csv('../data/KS_IV_example.csv')
print "数据规模：", df.shape

value_counts = df['category'].value_counts()
good_total = value_counts[0]
bad_total = value_counts[1]
total = good_total + bad_total
print '好账户共有：', good_total
print '坏账户共有：', bad_total

# 1.2 按照信用评分score对账户进行分组
# 
# - 要求
#     - 每组都至少有10%的用户（最后一组例外）
#     - 同一score的账户必须分入同一组

# 按照score值重新排序df
df_temp = df.sort_values(by='score')
df_sorted = df_temp.reset_index(level=0, drop=True)
bin_size = int(math.ceil(total * 0.1))
bins = []  # 记录每组最后一个账户
bin_size_list = []  # 记录每组账户个数
num_bins = 0
i = 0
start_index = 0
while i < total:
    end_index = start_index + bin_size - 1
    if end_index >= total - 1:
    # 最后一组，直接分组
        end_index = total - 1
    else:
    # 非最后一组，查看当前组内最后一个账户，是否与下个账户score相同。如果相同，则将下个账户分入当前组
        while end_index + 1 <= total - 1 and df_sorted.ix[end_index]['score'] == df_sorted.ix[end_index + 1]['score']:
            end_index = end_index + 1
    bins.append(end_index)
    bin_size_list.append(end_index-start_index)
    num_bins = num_bins + 1
    start_index = end_index + 1
    i = end_index + 1

# 1.3 计算KS和IV值

cum_good_ratio = 0
cum_bad_ratio = 0
cum_good_ratio_list = [0]
cum_bad_ratio_list = [0]

IV = 0
KS = 0
start_index = 0

i = 0
while i < num_bins:
    s1 = df_sorted[start_index:(bins[i] + 1)]
    s2 = s1[s1['category'] == 0]
    s3 = s1[s1['category'] == 1]
    good_in_bin = s2.index.size
    bad_in_bin = s3.index.size
    good_ratio_in_bin = good_in_bin / good_total
    bad_ratio_in_bin = bad_in_bin / bad_total
    cum_good_ratio = cum_good_ratio + good_ratio_in_bin
    cum_bad_ratio = cum_bad_ratio + bad_ratio_in_bin
    cum_good_ratio_list.append(cum_good_ratio)
    cum_bad_ratio_list.append(cum_bad_ratio)
    margin = abs(cum_good_ratio - cum_bad_ratio)
    if (margin > KS):
        KS = margin
    iv = (good_ratio_in_bin - bad_ratio_in_bin) * math.log(good_ratio_in_bin / bad_ratio_in_bin)
    IV = IV + iv
    start_index = bins[i] + 1
    i= i + 1

print 'KS: ',round(KS * 100, 1),'%'
print 'IV: ',IV

bin_ratio = [0]+[i*1.0/total for i in bin_size_list]
pylab.figure()
pylab.plot(range(len(cum_good_ratio_list)), cum_good_ratio_list, '-o',label='good')
pylab.plot(range(len(cum_bad_ratio_list)), cum_bad_ratio_list, '-o',label='bad')
pylab.legend(loc='upper left')
pylab.bar(range(len(bin_ratio)),bin_ratio)
pylab.ylabel("cum ratio")
pylab.xlabel("bin")
pylab.title('KS = '+str(round(KS * 100, 1))+"%")
pylab.savefig('../output/KS_example.png')

# 2. 计算数据集中所有变量的KS和IV值

def get_KS_IV(category, score):
    """category and score are both lists.
       return the KS and IV value.
    """
    
    cur_df = pd.DataFrame(zip(category,score),columns=['category','feature_score'])
    cur_df = cur_df.sort_values(by='feature_score')
    cur_df = cur_df.reset_index(level=0, drop=True)   
    value_counts = cur_df['category'].value_counts()
    good_total = value_counts[0]
    bad_total = value_counts[1]
    total = good_total + bad_total   
    bin_size = int(math.ceil(total * 0.1))
    bins = []# 记录每组最后一个账户
    num_bins = 0
    i = 0
    start_index = 0
    while i < total:
        end_index = start_index + bin_size - 1
        if end_index >= total - 1:
            # 最后一组，直接分组
            end_index = total - 1
        else:
            # 非最后一组，查看当前组内最后一个账户，是否与下个账户score相同。如果相同，则将下个账户分入当前组
            while end_index + 1 <= total - 1 and cur_df.ix[end_index]['feature_score'] == cur_df.ix[end_index + 1]['feature_score']:
                end_index = end_index + 1
        bins.append(end_index)
        num_bins = num_bins + 1
        start_index = end_index + 1
        i = end_index + 1   
    cum_good_ratio = 0
    cum_bad_ratio = 0
    start_index = 0
    IV = 0
    KS = 0
    i = 0
    while i < num_bins:
        s1 = cur_df[start_index:(bins[i] + 1)]
        s2 = s1[s1['category'] == 0]
        s3 = s1[s1['category'] == 1]
        good_in_bin = s2.index.size
        bad_in_bin = s3.index.size
        good_ratio_in_bin = good_in_bin / good_total+0.01
        bad_ratio_in_bin = bad_in_bin / bad_total+0.01
        cum_good_ratio = cum_good_ratio + good_ratio_in_bin
        cum_bad_ratio = cum_bad_ratio + bad_ratio_in_bin
        margin = abs(cum_good_ratio - cum_bad_ratio)
        if (margin > KS):
            KS = margin
        iv = (good_ratio_in_bin - bad_ratio_in_bin) * math.log(good_ratio_in_bin / bad_ratio_in_bin)
        IV = IV + iv
        start_index = bins[i] + 1
        i= i + 1
    return KS,IV


def get_KS_IV_features(category,feature_df):
    """categoty is the list to indicate whether the account is good. 
       feature_df is a data frame.
       return the KS and IV value lists.
    """
    KS_IV = []
    for feature in feature_df.columns:
        cur_KS, cur_IV = get_KS_IV(category,feature_df[feature])
        KS_IV.append([cur_KS, cur_IV])
        print '计算完毕:', feature
    return KS_IV


# 2.1 读数据
# 
# - data_clean.csv是1_preprocess处理后的数据
#     - 每行都表示一个借款账户
#     - loan_status = 0/1， 表示账户的好/坏
# - LCDataDictionary.csv是Leng Club数据中的变量含义
# - Completeness.csv是各变量缺失比例的数据

print '\n====================================='
print '计算数据集的KS和IV值'
print '====================================='

print '开始读取数据'
df = pd.read_csv('../data/data_clean.csv')
dict_df = pd.read_csv('../data/LCDataDictionary_clean.csv')
dict_df = dict_df.set_index('feature')
comp_df = pd.read_csv('../output/Completeness.csv')
comp_df = comp_df.set_index('feature')

print '开始计算KS和IV'
features = [i for i in df.columns if i not in ['loan_status','issue_d']]
KS_IV = get_KS_IV_features(df.loan_status, df[features])
KS_IV_df = pd.DataFrame(KS_IV, columns = ['KS','IV'],index = features)
KS_IV_df.index.name='feature'

show_features = ['delinq_2yrs','fico_range_low','fico_range_high','inq_last_6mths','mths_since_last_record']
show_KS_IV_df=KS_IV_df.loc[show_features,]
description_list = []
for feature in show_features:
    description_list.append(dict_df.loc[feature,'Description'])
show_KS_IV_df['feature desctiption'] = description_list
show_KS_IV_df = pd.concat([comp_df.loc[show_features,],show_KS_IV_df],axis=1)
print show_KS_IV_df

plt.figure()
show_KS_IV_df['KS'].plot.barh()
plt.title('KS of Different Features')
plt.tight_layout()
plt.savefig('../output/KS.png')
plt.figure()
show_KS_IV_df['IV'].plot.barh()
plt.title('IV of Different Features')
plt.tight_layout()
plt.savefig('../output/IV.png')
plt.figure()
show_KS_IV_df[['KS','IV']].plot.barh()
plt.title('Effectiveness of Different Features')
plt.tight_layout()
plt.savefig('../output/Effectiveness.png')
KS_IV_df.to_csv('../output/Effectiveness.csv')

print '\n请查看output中和KS、IV有关的图片和Effectiveness.csv文件'





