
# coding: utf-8

# # 构建Stacking预测模型
# - 目标：    
#    - 根据所有特征的完整性（ nan_ratio%）,有效性（KS/IV）和稳定性（PSI）来选取模型训练特征。
#    - 建立风控模型，对于测试集data_test.csv 中的每一条账户数据，预测其坏账（或成为坏账户）的概率（[0, 1]区间上的一个数）。

# In[1]:

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import tree  # We just train a decision tree classifier as an example. You can use whatever model you want.
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeCV,Ridge
from sklearn.cross_validation import KFold
from sklearn import *
import re


#  # 1. 读取数据
# - data_all.csv是Lending Club 2015年的1月~12月借贷数据，做训练用
#     - 每行都表示一个借款账户
#     - loan_status = 0/1， 表示账户的好/坏
#     - 除了“loan_status”以外，共计85 个字段均为借贷账户的特征
# - data_test.csv是lending club 2016年第一季度的数据，做测试用，内容同上
# - Comp_Effe_Stab.csv 存储的是前两次作业计算出的不同特征的nan_ratio，KS/IV以及不同月份的PSI值

# In[2]:

raw_df_train = pd.read_csv('../data/data_all.csv') 
df_test = pd.read_csv('../data/data_test.csv')
df_choose = pd.read_csv('../output/Comp_Effe_Stab.csv')

print 'Number of samples: train %d, test %d' % (raw_df_train.shape[0], df_test.shape[0])
print 'Number of features: %d' % (raw_df_train.shape[1])
df_choose.head()


# # 2. 特征选择
# - nan_ratio表示空缺值所占的比例，KS/IV表示有效性，即区分好人和坏人的能力，PSI表示特征的稳定性
# - 在特征选择时，有以下几步：
#     - 删除非数值型特征和对识别欺诈无意义的特征（如id, member_id等），得到data_clean.csv（见1_preprocess.ipynb)
#     - 选择data_clean.csv中任何两个月份的PSI值小于0.1的特征
#     - 将data_clean.csv中计算出的nan_ratio大于一定阈值的特征删去，因为nan_ratio过大，表示该特征的数据缺失太多
#     - 尽量选择KS/IV值较大的特征，但是本数据集算出特征的KS/IV值相差不大，因此在本例中这部分不做重点考虑
#     
#   注：选择PSI小于0.1的原因
#   ![](../others/PSI的含义.bmp)

# In[3]:

"""选择data_clean.csv中任何两个月份的PSI值小于0.1的特征,得到集合PSI_features
"""
m_df_choose = df_choose.shape[0]  
n_df_choose = df_choose.shape[1]
PSI_features = []
for i in range(m_df_choose):
    for j in range(5,n_df_choose):  #"5"表示PSI_1_2所在的列
        if (df_choose.loc[i][j] < 0.1):
            if j == n_df_choose - 1:
                PSI_features.append(df_choose.feature[i])
"""将data_clean.csv中计算出的nan_ratio大于一定阈值的特征删去，得到集合nan_feature
"""
df_del = df_choose[df_choose.nan_ratio < 0.05]  #将nan_ratio > 0.05的缺失严重的特征删掉
nan_feature =  list(df_del.feature)
"""求PSI_features和nan_feature两个集合的交集"""
features = list(set(PSI_features).intersection(set(nan_feature)))


# In[4]:

df_train = raw_df_train[features + ["loan_status"]]

# Simple strategy to fill the nan term
df_train = df_train.fillna(value=0, inplace=False)

print 'Number of samples:', df_train.shape[0], df_test.shape[0]
sample_size_dict = dict(pd.value_counts(df_train['loan_status']))
print "Negative sample size:%d,\nPositive sample size:%d\nImbalance ratio:%.3f"     % (sample_size_dict[0], sample_size_dict[1], float(sample_size_dict[1])/sample_size_dict[0])
df_train.head(3)


# # 3. 数据预处理——标准化

# - 对训练数据标准化

# In[5]:

sscaler = StandardScaler() # StandardScaler from sklearn.preprocessing
sscaler.fit(df_train[features]) # fit training data to get mean and variance of each feature term
train_matrix = sscaler.transform(df_train[features]) # transform training data to standardized vectors

train_labels = np.array(df_train['loan_status'])
print train_matrix.shape, train_labels.shape


# - 用相同的均值和方差对测试数据标准化

# In[6]:

df_test = df_test.fillna(value=0, inplace=False) # simply fill the nan value with zero
test_matrix = sscaler.transform(df_test[features]) # standardize test data
test_labels = np.array(df_test['loan_status'])
print test_matrix.shape, test_labels.shape
df_test[features].head(3)


# # 4. Stacking模型
# - 训练一个模型来组合其他各个模型。首先先训练多个不同的模型，然后再以之前训练的各个模型的输出为输入来训练一个模型，以得到一个最终的输出。

# ## 4.1 构造扩展类
# - 类SklearnHelper能够拓展所有Sklearn分类器的内置方法（包括train, predict and fit等），使得在第一级调用4种模型方法分类器时不会显得冗余（[见参考
# ](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)）

# In[7]:

# Some useful parameters which will come in handy later on
ntrain = df_train.shape[0]
ntest = df_test[features].shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)[:, 1]
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_) 


# ## 4.2 K-折交叉验证(k-fold CrossValidation)
# - 生成训练集和测试集的预测标签,[见参考](http://blog.csdn.net/yc1203968305/article/details/73526615)
#   ![](../others/Stacking.png)

# In[8]:

def get_oof(clf, x_train, y_train, test_matrix,test_labels):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict_proba(x_te)
        oof_test_skf[i, :] = clf.predict_proba(test_matrix)
        
    oof_test[:] = oof_test_skf.mean(axis=0)
    auc_score = metrics.roc_auc_score(y_true=test_labels, y_score=oof_test[:])
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1),auc_score


# ## 4.3 创建第一级分类器
# - Random Forest classifier
# - AdaBoost classifier
# - Neural_network MLP classifier
# - GradientBoosting classifier

# In[20]:

# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 100,
     'criterion': 'entropy', 
    'max_depth': 11,
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 200,
    'learning_rate' : 0.75
}

# Neural_network MLP classifier
mlp_params = {
    'hidden_layer_sizes': (150),
    'solver': 'adam',
    'activation': 'logistic',
    'alpha': 0.0001,
    'verbose': 1,
    'learning_rate_init': 0.01,
    'warm_start': True,
}

# Gradientboosting classifier
gb_params = {
    'learning_rate':0.01,
    'n_estimators': 1200,
    'max_depth': 9, 
    'min_samples_split': 60,
    'random_state': 10,
    'subsample': 0.85,
    'max_features':7,
    'warm_start': True,
}


# In[21]:

# 通过类SklearnHelper创建4个models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
mlp = SklearnHelper(clf=neural_network.MLPClassifier, seed=SEED, params=mlp_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)


# ## 4.4 第一级分类器的预测标签输出
# - XX_oof_train和XX_oof_test分别表示第一级分类器的训练集和测试集的预测标签输出

# In[29]:

# Create our OOF train and test predictions. These base results will be used as new features
rf_oof_train, rf_oof_test, rf_auc_score = get_oof(rf, train_matrix,train_labels, test_matrix,test_labels) # Random Forest
print("Random Forest is complete")
ada_oof_train, ada_oof_test, ada_auc_score = get_oof(ada,train_matrix,train_labels, test_matrix,test_labels) # AdaBoost 
print("AdaBoost  is complete")
mlp_oof_train, mlp_oof_test, mlp_auc_score = get_oof(mlp,train_matrix,train_labels, test_matrix,test_labels) # Neural_network MLP classifier
print("Neural_network MLP  is complete")
gb_oof_train, gb_oof_test, gb_auc_score = get_oof(gb,train_matrix,train_labels, test_matrix,test_labels) # GradientBoosting classifier
print("GradientBoosting is complete")


# In[30]:

print "Random Forest模型的AUC值:%.5f \nAdaBoost模型的AUC值:%.5f \nNeural_network MLP模型的AUC值:%.5f \nGradientBoosting模型的AUC值:%.5f"     % (rf_auc_score, ada_auc_score,mlp_auc_score,gb_auc_score)


# ## 4.5 合并第一级分类器的预测标签输出

# In[36]:

x_train = np.concatenate(( rf_oof_train, ada_oof_train, mlp_oof_train,gb_oof_train), axis=1)
x_test = np.concatenate(( rf_oof_test, ada_oof_test, mlp_oof_test,gb_oof_test), axis=1)


# ## 4.6 第一级各分类器方法的相关程度
# corr() 相关系数，默认皮尔森 0<|r|<1表示存在不同程度线性相关：
# - 0.8-1.0 极强相关
# - 0.6-0.8 强相关
# - 0.4-0.6 中等程度相关
# - 0.2-0.4 弱相关
# - 0.0-0.2 极弱相关或无相关

# In[32]:

df = pd.DataFrame(x_train, columns= ['rf','ada','mlp','gb'])
df_test = pd.DataFrame(x_test, columns= ['rf','ada','mlp','gb'])
df.corr()


# ## 4.7 训练第二级模型
# - 第二级模型选用的是岭回归模型方法
# - 岭回归的目标函数：
# ![](../others/函数.png)

# In[87]:

combinerModel = Ridge(alpha = 5000, fit_intercept=False)
combinerModel.fit(x_train, train_labels) 
print "Random Forest模型对最终预测结果的影响程度:%.5f \nAdaBoost模型对最终预测结果的影响程度:%.5f \nNeural_network MLP模型对最终预测结果的影响程度:%.5f \nGradientBoosting模型对最终预测结果的影响程度:%.5f"     % (combinerModel.coef_[0], combinerModel.coef_[1], combinerModel.coef_[2], combinerModel.coef_[3])
test_predictions = combinerModel.predict(x_test)
Stacking_auc_score = metrics.roc_auc_score(y_true=test_labels, y_score=test_predictions)
print "Stacking模型后最终的AUC值:%.5f " % (Stacking_auc_score)


# In[ ]:



