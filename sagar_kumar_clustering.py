#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


customers=pd.read_csv("F:/project/Zeotap/Customers.csv")
products=pd.read_csv("F:/project/Zeotap/Products.csv")
transactions=pd.read_csv("F:/project/Zeotap/Transactions.csv")


# In[3]:


#Preparing the data
#Aggregating transaction data by CustomerID
transaction_data = transactions.groupby("CustomerID").agg(
    total_spent=("TotalValue", "sum"),
    purchase_frequency=("TransactionID", "count"),
    avg_transaction_value=("TotalValue", "mean")
).reset_index()


# In[4]:


#Merging customer profile data with transaction data
customer_data = customers.merge(transaction_data, on="CustomerID")


# In[5]:


#Feature engineering:One-hot encode categorical columns
encoder=OneHotEncoder(sparse=False)
region_encoded=encoder.fit_transform(customer_data[["Region"]])
region_encoded_df=pd.DataFrame(region_encoded,columns=encoder.get_feature_names_out(["Region"]))


# In[7]:


# Combine profile and transaction features
features=pd.concat([customer_data[["total_spent","purchase_frequency","avg_transaction_value"]],region_encoded_df],axis=1)


# In[8]:


#Scaling the data (normalize values)
scaler=MinMaxScaler()
scaled_features=scaler.fit_transform(features)


# In[9]:


#Applying K-Means clustering
kmeans=KMeans(n_clusters=3, random_state=42)
customer_data["Cluster"]=kmeans.fit_predict(scaled_features)


# In[10]:


#Evaluating clustering using Davies-Bouldin Index
db_index=davies_bouldin_score(scaled_features, customer_data["Cluster"])


# In[11]:


#Visualizing the clusters (using PCA for dimensionality reduction)
pca=PCA(n_components=2)
principal_components=pca.fit_transform(scaled_features)
customer_data["PC1"]=principal_components[:, 0]
customer_data["PC2"]=principal_components[:, 1]


# In[12]:


#Plotting the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x="PC1",y="PC2",hue="Cluster",data=customer_data, palette="Set1",s=100,alpha=0.7)
plt.title("Customer Segmentation using K-Means Clustering")
plt.show()


# In[13]:


#Output the clustering results and evaluation metrics
print(f"Number of clusters: 3")
print(f"Davies-Bouldin Index: {db_index:.4f}")


# In[14]:


customer_data[["CustomerID", "Cluster"]].to_csv("Customer_Segmentation.csv", index=False)

