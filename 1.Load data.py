
# coding: utf-8

# In[ ]:


import pandas as pd
from pandas.io.json import json_normalize
import os
import json

import numpy as np
import re
import csv
import sys
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


root_path = "/Users/wangjiabin/Documents/data/impression-2018"


# In[ ]:


def load_data(path):
    
    df = pd.DataFrame()
    name_list=[]
    
    for path,dir_list,file_list in os.walk(path):         
        name_list.append(file_list)
        
        for file_name in file_list:
            if '.DS_Store' not in file_name.split('/')[-1]:
                df = df.append(pd.read_json(path+'/'+file_name,lines=True),ignore_index=True)
        
    meta = json_normalize(df.meta)
    univ = json_normalize(df.universal)
    df = df.join([meta,univ])
    return df


# In[ ]:


data = load_data(root_path)


# In[ ]:



def create_features(df,enforce_cols=None):
    
    #df = df.set_index(['MediaGammaImpression']) 
    df = df.drop(['cookie','meta','recommendation.items','universal','basket.line_items'],axis=1)
    
    
    df['screen.size'] = df['screen.height'] * df['screen.width']
    df['screen.size'] = df['screen.size'].astype(str)
    
    df['basket.item_count'] = df['basket.item_count'].replace(np.nan,'0')
    df['basket.item_count'] = df['basket.item_count'].astype(int)
    
    str_df = df.loc[:, df.dtypes == np.object].join(df.loc[:, df.dtypes == np.bool])
    sta_df = df.loc[:, df.dtypes == np.float64].join(df.loc[:, df.dtypes == np.int64])
    sta_df = sta_df.join(df['MediaGammaImpression'])
   
    str_df = str_df.replace(np.nan,'')
    str_df = str_df.astype(str)
    str_df['navigator.language'] = str_df['navigator.language'].apply(lambda x: x.upper())
    

    str_df = str_df.groupby('MediaGammaImpression').agg(lambda x: ''.join(set(x)))
    
    sta_df = df.groupby('MediaGammaImpression').agg({'basket.shipping_cost': 'sum',
                                                     'basket.subtotal': 'max',
                                                     'basket.tax': 'sum',
                                                     'basket.total': 'sum',
                                                     'product.unit_sale_price': 'sum',
                                                     'basket.item_count': 'max' })
    
    
    df = str_df.join(sta_df)
    
    df = df.reset_index(level='MediaGammaImpression')
    df = df.loc[df['MediaGammaImpression'].apply(lambda x: x != '')]
    df = df.set_index(['MediaGammaImpression']) 
    
    return str_df,sta_df,df
        


# In[ ]:


def pre_process_data(df, enforce_cols=None):
    #df.fillna(0, inplace=True)
    
    df['initial'] = df['initial'].apply(lambda x: 'False' if x == 'False'
                                    else('New' if 'TrueFalse' in x 
                                         else 'True'))
    
   
    df['document.referrer'] = df['document.referrer'].apply(lambda x: 
        'both' if 'petspyjamas.com/product' in x and 'petspyjamas.com/travel' in x
                    else('travel' if 'petspyjamas.com/travel' in x and  'petspyjamas.com/product' not in x 
                         else ('product' if 'petspyjamas.com/travel' not in x and  'petspyjamas.com/product' in x 
                               else 'others')))
    
    
    #df['document.referrer'] = df['document.referrer'].apply(lambda x: 'product' if 'petspyjamas.com/product' in x 
    #                                                        else('travel' if 'petspyjamas.com/travel' in x 
    #                                                             else('pet' if 'petspyjamas.com/pet' in x 
    #                                                                  else 'others'))) 
    df['href'] = df['href'].apply(lambda x: 'product' if 'petspyjamas.com/product' in x 
                                               else('travel' if 'petspyjamas.com/travel' in x 
                                                    else('pet' if 'petspyjamas.com/pet' in x 
                                                         else 'others'))) 

    #
    #document.title
    df['document.title'] = df['document.title'].apply(lambda x: 'both' if 'Shop' in x and 'Dog-friendly' in x
                                                      else('Shop' if 'Shop' in x and  'Dog-friendly' not in x 
                                                           else ('Dog-friendly' if 'Dog-friendly' in x and  'Shop' not in x
                                                                 else 'others')))
    
    
    df['page.location'] = df['page.location'].apply(lambda x: 'no_location' if x ==''  else 'location') 

    
    #navigator.platform
    
    df['navigator.platform'] = df['navigator.platform'].apply(lambda x: 'Linux' if 'Linux' in x
                                                      else('Win' if 'Win' in x  else x))
    
    #basket.currency
    df['basket.currency'] = df['basket.currency'].apply(lambda x: 'pay' if 'GBP' in x else 'not_pay')
    
    
    #product.url
    df['product.url'] = df['product.url'].apply(lambda x: 'product' if 'product' in x else 'others')
    
    #basket.total
    df.ix[df['basket.total'].isnull(), 'baskettotal'] = 0
    df.ix[df['basket.total'].between(1, 100), 'baskettotal'] = 1
    df.ix[df['basket.total'].between(100, 250), 'baskettotal'] = 2
    df.ix[df['basket.total'].between(250, 500), 'baskettotal'] = 3
    df.ix[df['basket.total']>500, 'baskettotal'] = 3
    
    #baskettotal#
    df.ix[df['basket.subtotal'].isnull(), 'basketsubtotal'] = 0
    df.ix[df['basket.subtotal'].between(1, 15), 'basketsubtotal'] = 1
    df.ix[df['basket.subtotal'].between(15, 50), 'basketsubtotal'] = 2
    df.ix[df['basket.subtotal'].between(50, 200), 'basketsubtotal'] = 3
    df.ix[df['basket.subtotal']>200, 'basketsubtotal'] = 3

    return df


# In[ ]:



def create_dummy(df, enforce_cols=None):
   
    # create dummy variables for categoricals
   # df.fillna(0, inplace=True)
    
    
    df_all = df[['document.referrer','document.title','href',
            'navigator.appCodeName','navigator.appName',
            'navigator.language','navigator.platform','navigator.product',
            'basket.currency','basket.item_count','basket.coupon',
            'page.location','page.category','page.type','product.category',
            'product.currency','product.url',
            'initial','navigator.cookieEnabled',
            'basket.subtotal','basket.total','baskettotal','basketsubtotal']]
    
    df = pd.get_dummies(df_all,dummy_na = True,
                        columns=['document.referrer','document.title','href',
                                 'navigator.appCodeName','navigator.appName',
                                 'navigator.language','navigator.platform','navigator.product',
                                 'basket.currency','basket.coupon',
                                 'page.location','page.category','page.type','product.category',
                                 'product.currency','product.url',
                                 'initial','navigator.cookieEnabled'])

    
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})
    
    df.fillna(0, inplace=True)
    return df_all,df



# In[ ]:


data = load_data(root_path)
str_df,sta_df,group_train = create_features(data)
group_train_pre = pre_process_data(group_train)
df_all,data_set = create_dummy(group_train_pre)  # data_all is user for clustering


# In[ ]:


df_all.to_csv("data_for_clustering.csv")
group_train.to_csv("group_for_nlp.csv")
#df_all.reset_index(level=['MediaGammaImpression'])

