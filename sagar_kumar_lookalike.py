#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


customers=pd.read_csv("F:/project/Zeotap/Customers.csv")
products=pd.read_csv("F:/project/Zeotap/Products.csv")
transactions=pd.read_csv("F:/project/Zeotap/Transactions.csv")


# In[4]:


#Merge datasets
data=transactions.merge(customers,on="CustomerID").merge(products,on="ProductID")


# In[5]:


#Aggregating features for customers
customer_features = data.groupby("CustomerID").agg(
    total_spent=("TotalValue","sum"),
    avg_purchase_value=("TotalValue","mean"),
    total_quantity=("Quantity","sum"),
    favorite_category=("Category",lambda x: x.mode()[0]),
    region=("Region","first"),
    signup_date=("SignupDate","first")
).reset_index()


# In[6]:


#Encoding categorical features
encoder=OneHotEncoder()
encoded_categories=encoder.fit_transform(customer_features[["region","favorite_category"]]).toarray()


# In[7]:


#Normalizing numerical features
scaler = MinMaxScaler()
numerical_features = scaler.fit_transform(
    customer_features[["total_spent", "avg_purchase_value", "total_quantity"]])


# In[8]:


#Combining all features
final_features = pd.concat([
    pd.DataFrame(numerical_features),
    pd.DataFrame(encoded_categories)
], axis=1)


# In[9]:


#Calculating similarity
similarity_matrix=cosine_similarity(final_features)


# In[10]:


#Generating top 3 lookalikes for the first 20 customers
lookalike_map = {}
for idx in range(20):  # First 20 customers
    customer_id = customer_features.loc[idx, "CustomerID"]
    similarities = list(enumerate(similarity_matrix[idx]))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[1:4]  # Top 3 (excluding self)
    lookalike_map[customer_id] = [(customer_features.loc[sim[0], "CustomerID"], round(sim[1], 2)) for sim in similarities]


# In[11]:


# Step 8: Save the lookalike map as CSV
lookalike_df=pd.DataFrame({
    "cust_id":list(lookalike_map.keys()),
    "lookalikes":[str(lookalike_map[cust]) for cust in lookalike_map]
})
lookalike_df.to_csv("Lookalike.csv",index=False)

print("Lookalike.csv generated successfully")

