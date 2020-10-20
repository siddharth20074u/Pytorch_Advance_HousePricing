#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:11:58 2020

@author: siddharthsmac
"""

import pandas as pd
import numpy as np

df = pd.read_csv('/users/siddharthsmac/downloads/houseprice.csv', usecols = ['SalePrice', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'YearBuilt', 'LotShape', '1stFlrSF', '2ndFlrSF']).dropna()

for i in df.columns:
    print('Column name {} and unique values are {}'.format(i, len(df[i].unique())))
    
import datetime

df['Total_Years'] = datetime.datetime.now().year - df['YearBuilt']
df.drop('YearBuilt', axis = 1, inplace = True)

cat_features = ['MSSubClass', 'MSZoning', 'Street', 'LotShape']
out_features = 'SalePrice'

from sklearn.preprocessing import LabelEncoder

lbl_encoders = {}
for feature in cat_features:
    lbl_encoders[feature] = LabelEncoder()
    df[feature] = lbl_encoders[feature].fit_transform(df[feature])

cat_features = np.stack([df['MSSubClass'], df['MSZoning'], df['Street'], df['LotShape']], 1)

import torch

cat_features = torch.tensor(cat_features, dtype = torch.int64)

cont_features = []
for i in df.columns:
    if i in ['MSSubClass', 'MSZoning', 'LotShape', 'SalePrice']:
        pass
    else:
        cont_features.append(i)

cont_values = np.stack([df[i].values for i in cont_features], axis = 1)

cont_values = torch.tensor(cont_values, dtype = torch.float)

y = torch.tensor(df['SalePrice'].values, dtype = torch.float).reshape(-1, 1)

# Embedding done only for categorical features

cat_dims = [len(df[col].unique()) for col in ['MSSubClass', 'MSZoning', 'Street', 'LotShape']]

# Thumbs rule for categorical embedding based on article in fast.ai 

embedding_dim = [(x, min(50, (x+1)//2)) for x in cat_dims] 

import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, embedding_dim, n_cont, out_sz, layers, p = 0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(inp, out) for inp, out in embedding_dim])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        layerlist = []
        n_emb = sum(out for inp, out in embedding_dim)
        n_in = n_emb + n_cont        
        
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace = True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        
        layerlist.append(nn.Linear(layers[-1], out_sz))
        self.layers = nn.Sequential(*layerlist)
        
    def forward(self, x_cat, x_cont):
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))
            
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x
    
torch.manual_seed(100)

model = FeedForwardNN(embedding_dim, len(cont_features), 1, [100, 50], p = 0.4)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)        
        
batch_size = 1200
test_size = int(batch_size*0.15)
train_categorical = cat_features[: batch_size-test_size]
test_categorical = cat_features[batch_size-test_size : batch_size]        
train_cont = cont_values[: batch_size-test_size]
test_cont = cont_values[batch_size-test_size : batch_size] 
y_train = y[: batch_size-test_size]
y_test = y[batch_size-test_size : batch_size] 

epochs = 5000
final_losses = []
for i in range(epochs):
    i = i+1
    y_pred = model(train_categorical, train_cont)
    loss = torch.sqrt(loss_function(y_pred, y_train))
    final_losses.append(loss)
    if i%10 == 1:
        print('Epoch number: {} and the loss: {}'.format(i, loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
import matplotlib.pyplot as plt

plt.plot(range(epochs), final_losses)
plt.ylabel('RMSE Loss')
plt.xlabel('Epochs')
plt.title('Loss across iterartion')
plt.show() 

# validate test data

y_pred = ''
with torch.no_grad():
    y_pred = model(test_categorical, test_cont)
    loss = torch.sqrt(loss_function(y_pred, y_test))
print('RMSE: {}'.format(loss))

torch.save(model.state_dict(), 'HousePricing.pt')  
