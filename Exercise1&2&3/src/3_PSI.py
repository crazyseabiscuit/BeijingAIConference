
# coding: utf-8

"""计算不同特征的PSI值，比较它们的稳定性
"""

from __future__ import division
import math
import numpy as np

import pandas as pd
from matplotlib import pylab
import matplotlib.pyplot as plt

# 1. 读数据
# data_clean.csv是1_preprocess处理后的数据
# 每行都表示一个借款账户
# issue_d表示申请贷款的月份
# LCDataDictionary.csv是Leng Club数据中的变量含义
# Completeness.csv是各变量缺失比例的数据
# Effectiveness.csv是各变量KS和IV的数据

df = pd.read_csv('../data/data_clean.csv')
# df = pd.read_csv('../data/PSI_6m.csv')
month_max = max(df.issue_d)

print "数据规模：", df.shape
print '一共'+str(month_max)+'个月的数据'

dict_df = pd.read_csv('../data/LCDataDictionary_clean.csv')
dict_df = dict_df.set_index('feature')
comp_df = pd.read_csv('../output/Completeness.csv')
comp_df = comp_df.set_index('feature')
effe_df = pd.read_csv('../output/Effectiveness.csv')
effe_df = effe_df.set_index('feature')

# 2. 计算tot_cur_bal特征在1月和6月的PSI值

print '====================================='
print '计算tot_cur_bal特征在1月和6月的PSI值'
print '====================================='

df_temp = df[['issue_d','tot_cur_bal']]
df_temp = df_temp[df_temp['issue_d'].isin([1,6])]
before_total = df_temp[df_temp['issue_d']==1].shape[0]
after_total = df_temp[df_temp['issue_d']==6].shape[0]
total = before_total + after_total
print '1月数据条数：', before_total
print '6月数据条数：', after_total

# 如果存在缺失值，则将没有tot_cur_bal特征的数据单独归为一组，计算其psi
df_null_index = np.isnan(df_temp['tot_cur_bal'])
df_null = df_temp.ix[df_null_index]
if len(df_null) > 0:
    before_in_bin = len(df_null[df_null['issue_d']==before_time])
    after_in_bin = len(df_null[df_null['issue_d']==after_time])
    befor_ratio_in_bin = before_in_bin / before_total + 0.01
    after_ratio_in_bin = after_in_bin / after_total + 0.01
    PSI  = (befor_ratio_in_bin - after_ratio_in_bin) * math.log(befor_ratio_in_bin / after_ratio_in_bin)
    total_no_nan = total - before_in_bin -after_in_bin
else:
    PSI = 0
    total_no_nan = total

#   去除缺失值后按照tot_cur_bal对数据进行排序和分组
# 
#   要求
#     - 每组都至少有10%的用户（最后一组例外）
#     - 同一score的账户必须分入同一组

df_temp.dropna(how='any')
df_temp = df_temp.sort_values(by='tot_cur_bal')
df_temp = df_temp.reset_index(level=0, drop=True)

bin_size = int(math.ceil(total * 0.1))

bins = []  # 记录每组最后一个账户

num_bins = 0

i = 0
start_index = 0
while i < total_no_nan:

    end_index = start_index + bin_size - 1
    if end_index >= total - 1:
    # 最后一组，直接分组
        end_index = total - 1
    else:
    # 非最后一组，查看当前组内最后一个账户，是否与下个账户tot_coll_amt特征值相同。如果相同，则将下个账户分入当前组
        while end_index + 1 <= total - 1 and df_temp.ix[end_index]['tot_cur_bal'] == df_temp.ix[end_index + 1]['tot_cur_bal']:
            end_index = end_index + 1

    bins.append(end_index)
    num_bins = num_bins + 1

    start_index = end_index + 1
    i = end_index + 1

# 计算PSI值

start_index = 0
i = 0
while i < num_bins:
    s1 = df_temp[start_index:(bins[i] + 1)]
    s2 = s1[s1['issue_d'] == 1]
    s3 = s1[s1['issue_d'] == 6]
    before_in_bin = s2.index.size 
    after_in_bin = s3.index.size
    befor_ratio_in_bin = before_in_bin / before_total + 0.01
    after_ratio_in_bin = after_in_bin / after_total + 0.01
    psi = (befor_ratio_in_bin - after_ratio_in_bin) * math.log(befor_ratio_in_bin / after_ratio_in_bin)
    PSI= PSI + psi
    start_index = bins[i] + 1
    i= i + 1

print 'PSI: ', PSI


# 2. 计算所有特征的PSI值，依次让2月同1月对比，3月同1月对比...6月同1月对比


def get_PSI(time,feature):
    """time and feature are both lists.
       return the PSI value.
    """
    df_temp = pd.DataFrame(zip(time,feature),columns=['issue_d','feature_score'])
    issue_d_values = sorted(list(set(time)))
    before_time = issue_d_values[0]
    after_time = issue_d_values[1]
    before_total = df_temp[df_temp['issue_d']==before_time].shape[0]
    after_total = df_temp[df_temp['issue_d']==after_time].shape[0]
    total = before_total + after_total
    
    df_null_index = np.isnan(df_temp['feature_score'])
    df_null = df_temp.ix[df_null_index]
    if len(df_null) > 0:
        before_in_bin = len(df_null[df_null['issue_d']==before_time])
        after_in_bin = len(df_null[df_null['issue_d']==after_time])
        befor_ratio_in_bin = before_in_bin / before_total + 0.01
        after_ratio_in_bin = after_in_bin / after_total + 0.01
        PSI  = (befor_ratio_in_bin - after_ratio_in_bin) * math.log(befor_ratio_in_bin / after_ratio_in_bin)
        total_no_nan = total - before_in_bin -after_in_bin
    else:
        PSI = 0
        total_no_nan = total
    
    df_temp = df_temp.dropna(how='any')
    df_temp = df_temp.sort_values(by='feature_score')
    df_temp = df_temp.reset_index(level=0, drop=True)

    
    bin_size = int(math.ceil(total_no_nan * 0.1))

    bins = []  # 记录每组最后一个账户
    num_bins = 0

    i = 0
    start_index = 0
    while i < total_no_nan:

        end_index = start_index + bin_size - 1
        if end_index >= total_no_nan - 1:
        # 最后一组，直接分组
            end_index = total_no_nan - 1
        else:
        # 非最后一组，查看当前组内最后一个账户，是否与下个账户feature_score特征值相同。如果相同，则将下个账户分入当前组
            while end_index + 1 <= total_no_nan - 1 and df_temp.ix[end_index]['feature_score'] == df_temp.ix[end_index + 1]['feature_score']:
                end_index = end_index + 1

        bins.append(end_index)
        num_bins = num_bins + 1

        start_index = end_index + 1
        i = end_index + 1
        
    start_index = 0
    PSI = 0
    i = 0
    
    while i < num_bins:
        s1 = df_temp[start_index:(bins[i] + 1)]
        s2 = s1[s1['issue_d'] == before_time]
        s3 = s1[s1['issue_d'] == after_time]

        before_in_bin = s2.index.size 
        after_in_bin = s3.index.size

        befor_ratio_in_bin = before_in_bin / before_total + 0.01
        after_ratio_in_bin = after_in_bin / after_total + 0.01

        psi = (befor_ratio_in_bin - after_ratio_in_bin) * math.log(befor_ratio_in_bin / after_ratio_in_bin)
        PSI= PSI + psi

        start_index = bins[i] + 1
        i= i + 1

    return PSI


def get_PSI_features(time, feature_df):
    """time is a list and feature_df is a data frame.
       return the PSI values of every feature in the feature_df.
    """    
    PSI = []
    for feature in feature_df.columns:
        cur_PSI = get_PSI(time,feature_df[feature])
        PSI.append(cur_PSI)
    return PSI


# 选取特征，计算不同月份同1月比较的PSI值
print '\n====================================='
print '开始计算多个特征在不同月份的PSI值'
print '====================================='

PSI_list = []
before_time = 1
feature_cols = [i for i in df.columns if i not in ['issue_d','loan_status']]


print '计算各月同1月比较的PSI值:'
for after_time in range(2,month_max+1):
    cur_df = df[df['issue_d'].isin([before_time, after_time])]
    PSI_list.append(get_PSI_features(cur_df['issue_d'],cur_df[feature_cols]))
    print '完成所有特征在1月和'+str(after_time)+'月的比较'
index_names = ['PSI_1_'+str(i) for i in range(2,month_max+1)]
PSI_df = pd.DataFrame(PSI_list, columns = feature_cols,index = index_names).T
PSI_df.index.name='feature'

show_features = ['delinq_2yrs','fico_range_low','fico_range_high','inq_last_6mths','mths_since_last_record']
pylab.figure(figsize=[8,5])
for feature in show_features:
    pylab.plot(range(2,month_max+1), PSI_df.loc[feature], '-o',label=feature)
pylab.legend(loc='upper left')
pylab.ylabel("PSI")
pylab.xlabel("Month")
pylab.ylim([0,0.01])
pylab.title('Stability of Different Features: Compared with Jan')
pylab.savefig('../output/Stability_compare_with_Jan.png')

print '\n计算各月同上月比较的PSI值:'
PSI_list2 = []
for after_time in range(2,month_max+1):
    before_time = after_time - 1
    cur_df = df[df['issue_d'].isin([before_time, after_time])]
    PSI_list2.append(get_PSI_features(cur_df['issue_d'],cur_df[feature_cols]))
    print '完成所有特征在'+str(before_time)+'月和'+str(after_time)+'月的比较'
index_names = ['PSI_'+str(i-1)+'_'+str(i) for i in range(2,month_max+1)]
PSI_df2 = pd.DataFrame(PSI_list2, columns = feature_cols,index = index_names).T
pylab.figure(figsize=[8,5])
for feature in show_features:
    pylab.plot(range(2,month_max+1), PSI_df2.loc[feature,], '-o',label=feature)
pylab.legend(loc='upper left')
pylab.ylabel("PSI")
pylab.xlabel("Month")
pylab.ylim([0,0.01])
pylab.title('Stability of Different Features: Compared with Last Month')
pylab.savefig('../output/Stability_compare_with_last_month.png')

stab_df = pd.concat([PSI_df,PSI_df2],axis=1)
stab_df.index_name = 'feature'
stab_df.to_csv('../output/Stability.csv')
pd.concat([dict_df.ix[feature_cols],comp_df,effe_df,stab_df], axis=1).\
          to_csv('../output/Comp_Effe_Stab.csv')

print '\n请查看output中的Stability.png和Comp_Effe_Stab.csv文件'



