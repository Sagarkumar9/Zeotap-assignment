#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


customers=pd.read_csv("F:/project/Zeotap/Customers.csv")
products=pd.read_csv("F:/project/Zeotap/Products.csv")
transactions=pd.read_csv("F:/project/Zeotap/Transactions.csv")


# In[3]:


print(customers.info(),"\n")
print(products.info(),"\n")
print(transactions.info(),"\n")


# In[13]:


#converting object to date
customers['SignupDate']=pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate']=pd.to_datetime(transactions['TransactionDate'])


# In[14]:


print("Customers - Data Types After Conversion:")
print(customers.dtypes, "\n")
print("Transactions - Data Types After Conversion:")
print(transactions.dtypes, "\n")


# In[12]:


print("Customers:",customers.duplicated().sum())
print("Products:",products.duplicated().sum())
print("Transactions:",transactions.duplicated().sum())


# In[6]:


#Univariate Analysis
print("Customer Regions:\n",customers['Region'].value_counts())
print("\nProduct Categories:\n", roducts['Category'].value_counts())
#Transaction Value Distribution
plt.figure(figsize=(8,5))
sns.histplot(transactions['TotalValue'],kde=True,bins=30,color='skyblue')
plt.title("Distribution of Transaction Values")
plt.xlabel("Transaction Value")
plt.ylabel("Frequency")
plt.show()


# In[7]:


# Top 10 Customers by Total Spending
top_customers = transactions.groupby('CustomerID')['TotalValue'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Customers by Total Spending:\n",top_customers)
top_customers.plot(kind='bar',title='Top 10 Customers by Spending',color='orange')
plt.ylabel("Total Spending")
plt.show()


# In[8]:


#Multivariate Analysis
#Correlation between numerical variables
correlation = transactions[['Quantity', 'Price','TotalValue']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# In[9]:


#Revenue by Region
region_revenue = transactions.merge(customers,on="CustomerID").groupby('Region')['TotalValue'].sum()
region_revenue.plot(kind='bar',title='Revenue by Region',color='green')
plt.ylabel("Revenue (USD)")
plt.show()


# In[10]:


#Monthly Revenue Trend
transactions['Month']=transactions['TransactionDate'].dt.to_period('M')
monthly_revenue=transactions.groupby('Month')['TotalValue'].sum()
monthly_revenue.plot(marker='o',title='Monthly Revenue Trend', color='purple')
plt.ylabel("Revenue (USD)")
plt.xlabel("Month")
plt.show()


# In[11]:


category_revenue = transactions.merge(products, on="ProductID").groupby('Category')['TotalValue'].sum()
category_revenue.plot(kind='bar', color='blue', figsize=(8, 5), title='Revenue by Product Category')
plt.ylabel("Revenue (USD)")
plt.xlabel("Category")
plt.show()


# In[ ]:




