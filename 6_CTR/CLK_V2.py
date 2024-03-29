# coding: utf-8

import numpy as np
import pandas as pd
import dask.dataframe as dd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sklearn

init_notebook_mode(connected=True)
matplotlib.style.use('ggplot')


date_parser = lambda x: pd.datetime.strptime(x, '%y%m%d%H')



data_types = {
    'id': np.str,
    'click': np.bool_,
    'hour': np.str,
    'C1': np.uint16,
    'banner_pos': np.uint16,
    'site_id': np.object,
    'site_domain': np.object,
    'site_category': np.object,
    'app_id': np.object,
    'app_domain': np.object,
    'app_category': np.object,
    'device_id': np.object,
    'device_ip': np.object,
    'device_model': np.object,
    'device_type': np.uint16,
    'device_conn_type': np.uint16,
    'C14': np.uint16,
    'C15': np.uint16,
    'C16': np.uint16,
    'C17': np.uint16,
    'C18': np.uint16,
    'C19': np.uint16,
    'C20': np.uint16,
    'C21': np.uint16
}




def load_train_data():
    train_df = pd.read_csv('ctr_train_sample.csv',
                           dtype=data_types,
                           parse_dates=['hour'],
                           date_parser=date_parser)
    return train_df


train_df = load_train_data()


train_df.info()
train_df.head(1)
train_df['click'].mean()


train_df.iloc[:, :12].head()


train_df.iloc[:, 12:].head()


train_df.shape[0] == train_df['id'].unique().shape[0]



def remove_id_col(df):
    df.drop('id', axis=1, inplace=True)
    return df


train_df = remove_id_col(train_df)


train_df.groupby('click').size().plot(kind='bar')
plt.show()

# In[16]:

train_df['click'].value_counts() / train_df.shape[0]


print(train_df.hour.describe())



def derive_time_features(df, start_hour=None, remove_original_feature=False):
    if start_hour is None:
        start_hour = df['hour'][0]

    df['hour_int'] = train_df['hour'].apply(
        lambda x: np.floor((x - start_hour) / np.timedelta64(1, 'h')).astype(np.uint16))
    df['day_week'] = train_df['hour'].apply(lambda x: x.dayofweek)
    df['hour_day'] = train_df['hour'].apply(lambda x: x.hour)

    if remove_original_feature:
        df.drop('hour', axis=1, inplace=True)

    return df, start_hour


# In[19]:

train_df, _ = derive_time_features(train_df)



train_df.groupby(['day_week', 'click']).size().unstack().plot(kind='bar', stacked=True, title="Days of the week")
plt.show()



train_df.groupby(['hour_day', 'click']).size().unstack().plot(kind='bar', stacked=True, title="Hours of the day")
plt.show()


train_df['banner_pos'].unique()



train_banner_pos_group_df = train_df.groupby(['banner_pos', 'click']).size().unstack()

# In[24]:

train_banner_pos_group_df.plot(kind='bar', stacked=True, title='Banner position')
plt.show()

# Positions 0 and 1 seem to be the most used ones. We have to make another plot to gain insight over the other variables:

# In[25]:

train_banner_pos_group_df.iloc[2:].plot(kind='bar', stacked=True, title='Banner position')
plt.show()

# Position 7 seems to be very good for positioning ads.
# These are the CTRs, normalized by banner position:

# In[26]:

train_banner_pos_group_df / train_df.shape[0]

# Now normalized by banner position:

# In[27]:

train_banner_pos_group_df.div(train_banner_pos_group_df.sum(axis=1), axis=0)

# Effectively, the CTR for position 7 is slightly above 32%. Positions 0, 1, 3 and 4 seem to be close contenders, with almost 20%, with the remainders slightly above 10%.

# ### Site-related features
# Regarding the site, we have the features id, domain and category:

# In[28]:

site_features = ['site_id', 'site_domain', 'site_category']

# In[29]:

print(train_df[site_features].describe())

# ### App-related features
# Regarding the app, we have the features id, domain and category:

# In[30]:

app_features = ['app_id', 'app_domain', 'app_category']

# In[31]:

print(train_df[app_features].describe())

# In[32]:

train_df['app_category'].value_counts().plot(kind='bar', title='App Category Histogram')
plt.show()

# In[33]:

train_app_category_group_df = train_df.groupby(['app_category', 'click']).size().unstack()

# In[34]:

train_app_category_group_df.div(train_app_category_group_df.sum(axis=1), axis=0).plot(kind='bar', stacked=True,
                                                                                      title="Intra-category CTR")
plt.show()

# ### Device-related features
# Regarding the device, we have the features id, ip, model, type and connection type:

# In[35]:

device_features = ['device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type']

# In[36]:

print(train_df[device_features].astype('object').describe())

# In[37]:

train_df.groupby(['device_type', 'click']).size().unstack().plot(kind='bar', stacked=True,
                                                                 title='Device type histogram')
plt.show()

# In[38]:

train_df.groupby(['device_conn_type', 'click']).size().unstack().plot(kind='bar', stacked=True,
                                                                      title='Device connection type histogram')
plt.show()

# In what regards the devices, it would be interesting to create a dataframe where the rows are device_ids, and the columns correspond to historic characteristics of that device usage, like which site categories it visited. This information could be plugged in each row, joining the dataframes by device_id.
# 
# Due to time constraints, I was not able to further develop this topic. For an example of a project of mine see this [link](http://nbviewer.jupyter.org/github/diogojapinto/banking-project/blob/master/Banking%20Project%20-%20Diogo.ipynb) (bear in mind it was almost 2 years ago).

# ### C1, C14-C21
# These are a couple of features whose identity was hidden, for annonimity reasons.

# In[39]:

annonym_features = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']

# In[40]:

print(train_df[annonym_features].astype('object').describe())

# In[41]:

train_df.groupby(['C1', 'click']).size().unstack().plot(kind='bar', stacked=True, title='C1 histogram')
plt.show()

# In[42]:

train_df.groupby(['C15', 'click']).size().unstack().plot(kind='bar', stacked=True, title='C15 histogram')
plt.show()

# In[43]:

train_df.groupby(['C16', 'click']).size().unstack().plot(kind='bar', stacked=True, title='C1 histogram')
plt.show()

# In[44]:

train_df.groupby(['C18', 'click']).size().unstack().plot(kind='bar', stacked=True, title='C18 histogram')
plt.show()

# ## Prediction task
# 
# Provided the knowledge gathered in the previous stage, lets now dive in the prediction task.
# Firstly, data must be prepared so that it can be fed to machine learning algorithms.

# ### Data preparation

# These are the features that will be used at our prediction task:

# In[45]:

features_mask = ['hour_int', 'day_week', 'hour_day', 'banner_pos', 'site_category']

# In[46]:

target_mask = 'click'

# Lets extract a sample from the data, to fasten computations:

# In[47]:

train_sample_df = train_df[features_mask + [target_mask]].sample(frac=0.01, random_state=42)


# We have to convert *site_category* into a format understandable by our classifiers.
# Additionally, *banner_pos* is currently represented as an integer. We do not want our classification methods to be influenced by the arbitrary ordering that features has.
# One-hot encoding will, therefore, be used:

# In[48]:

def one_hot_obj_features(df, features):
    new_df = pd.get_dummies(df, columns=features, sparse=True)
    return new_df


# In[49]:

train_sample_df = one_hot_obj_features(train_sample_df, ['site_category', 'banner_pos'])

# Before moving one, we should extract a test set, that will only be touched at the time we obtain the model we will use, in order to estimate its performance with real-world data.

# In[50]:

features_mask = np.array(train_sample_df.columns[train_sample_df.columns != target_mask].tolist())

# In[51]:

from sklearn.model_selection import train_test_split

# In[52]:

X_train, X_test, y_train, y_test = train_test_split(
    train_sample_df[features_mask].values,
    train_sample_df[target_mask].values,
    stratify=train_sample_df[target_mask],
    test_size=0.3,
    random_state=42
)

# ### Decision tree
# In order to gain some intuition on the patterns found in data that are behind the prediction decisions, we can model a decision tree and visualize it. This kind of model is naturally interpretable.

# In[65]:

from sklearn.tree import DecisionTreeClassifier

# In[66]:

dt_clf = DecisionTreeClassifier(min_samples_split=20, random_state=0, min_samples_leaf=2, max_depth=3,
                                class_weight='balanced')

# In[67]:
print("x_train shape")
print(X_train)
print("y_train shape")
print(y_train)
dt_clf.fit(X_train, y_train)

# Next follows a piece of code that enables us to visualize the obtained tree (this step requires installation of GraphViz):

# In[68]:

from sklearn import tree
from sklearn.externals.six import StringIO
import pydot as pydot
from IPython.display import Image


def display_tree(dtc_classifier):
    dot_data = StringIO()
    tree.export_graphviz(dtc_classifier, out_file=dot_data,
                         feature_names=features_mask,
                         class_names=target_mask,
                         filled=True,
                         rounded=True,
                         special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    return Image(graph[0].create_png())


# In[69]:

# display_tree(dt_clf)


# We are able to see that the site categories are recursively used to split the data. The tree also seeks to subset the time dimension into intervals to classify the data.
# 
# When analysing this tree, and due to the dataset dimension, it is important to remember that the tree height

# ### Gradient Boosting

# In[70]:

from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# In[71]:

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    stratify=y_train,
    test_size=0.3,
    random_state=42
)

# In[74]:

xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=[(X_valid, y_valid)])
y_pred = xgb_clf.predict(X_test)
predictions = [round(value) for value in y_pred]

# In[77]:

print(classification_report(y_test, predictions))


# ## Predict for new data
# Now is time to apply the classification algorithm to obtain predictions on unseen data.
# We have to prepare the test dataset as we have done to the training dataset:

# In[ ]:

def load_t_data():
    test_df = pd.read_csv('ctr_test_sample.csv',
                          dtype=data_types,
                          parse_dates=['hour'],
                          date_parser=date_parser)
    return test_df


# In[ ]:

test_df = load_t_data()


# In[ ]:
# 测试准确性
# 训练数据全部准确性
# 提高
# LR
# XGboost +LR

# precision = TP/(TP+FP)
# recall= TP/(TP+FN)
# accuracy = (TP+FN)/(TP+FP+FN)

