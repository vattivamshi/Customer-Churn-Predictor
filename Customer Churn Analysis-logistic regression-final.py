#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.ticker as mtick  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install matplotlib')


# In[3]:



telecom_data=pd.read_csv("C:\\Users\\VATTI VAMSHI\\Downloads\\WA_Fn-UseC_-Telco-Customer-Churn (1).csv")


# In[4]:


telecom_data.head()


# In[5]:


telecom_data.columns


# In[6]:


telecom_data.shape


# In[7]:


telecom_data.dtypes
#total charges should be int type instead of object


# In[8]:


telecom_data.describe() #descriptive statistics for numerical variables
#There is inconsistency in senior citizen 25-75-100 percentages as senior citizens is a categorical data and not numerical


# In[9]:


#Target variables plotted against various other variables

telecom_data['Churn'].value_counts().plot(kind='barh', figsize=(8, 6))
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variable", labelpad=14)
plt.title("Count of TARGET Variable per category", y=1.02);


# In[10]:




#Chcking the percentage of target variable elements for all the other variables

telecom_data['Churn'].value_counts()/len(telecom_data['Churn'])*100

#Data is imbalanced it doesn't represent the churners and non churners equally which may lead problems while building prediction models
#upsampling or downsampling needs to be done while creating a prediction model


# In[11]:


telecom_data.info(verbose=True)


# In[12]:


#Examining null values

null_value_columns=telecom_data.isnull().sum()


# In[13]:


null_value_columns


# In[14]:


#After gathering these insights we need to clean the data first

#We create a base data to make changes
telecom_base_data=telecom_data

#Then we start by removing null values

telecom_base_data.dropna() #inorder to drop null values(there are no null values in this case)


# In[15]:


#Convert columns into appropriate data types:
                                                      


# In[16]:


# Replace empty strings with NaN
telecom_base_data["TotalCharges"] = telecom_base_data["TotalCharges"].replace(' ', np.nan)

# Convert the column to numeric
telecom_base_data["TotalCharges"] = pd.to_numeric(telecom_base_data["TotalCharges"])

#or

telecom_base_data["TotalCharges"]=pd.to_numeric(telecom_base_data["TotalCharges"],errors='coerce')



# In[17]:


telecom_base_data.isnull().sum()


# In[18]:


#there are 11 null values in total charges after we converted the data type
telecom_base_data.dropna(how='any',inplace=True)


# In[19]:


telecom_base_data.isnull().sum()


# In[20]:


telecom_base_data["tenure"].max()


# In[21]:


#Lets convert continous data in tenure into value bins for easy analyses

labels=["{0}-{1}".format(i,i+11) for i in range(1,72,12)]

telecom_base_data["tenure_group"]=pd.cut(telecom_base_data["tenure"],range(1,80,12),right=False,labels=labels)
#Here right=False suggests that we do not include upper bound while categorisation


# In[22]:


telecom_base_data["tenure_group"].value_counts()


# In[23]:


telecom_base_data["tenure_group"].values


# In[24]:


#Next we remove the columns that we don't use for calculations

telecom_base_data.drop(columns=['customerID','tenure'],axis=1, inplace=True)


# In[25]:


telecom_base_data.head()


# In[26]:


#Exploratory Data Analysis
#Univariate Analysis


# In[27]:


#Univariate analysis for categorical Data
#We use count plot to see counts of different predictor categorical variables for the target variable churn


# In[28]:


for i,predictor in enumerate(telecom_base_data.drop(columns=['Churn','TotalCharges','MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=telecom_base_data,x=predictor,hue='Churn')


# In[29]:


#FOLLOWING INSIGHTS CAN BE DERIVED FROM THESE COUNT PLOTS:
 #1Gender has no effect on churn
 #2.Customers without partners have relatively more churnes compared to those with partners*
 #3.Customers without dependents have more churners compared to the one with dependents*
#4.High Churn for fibre optic internet service

#5 High Churn for no online security
#6 High Churn for no online backup
#7High churn for no device protection
#8 High Churn for no tech support
#9 High churn for month to month contract

#10 High churn for paperless billing

#11 High churn for electronic check payment method

#low churner group#
#1 Customer with a phone service have less churners
#2Customers without multiple lines have low churners
#Tenure Groups
#High churners for tenure group of 1 year i.e 1 to 12 months
#Low chuners for tenure group with 5+ years


# In[30]:


#Exploratory analysis for numerical variables
 #We are going to use kernel density plot inorder to asses the numerical variables instead of histograms to avoid bin size sensitivity and
 #to show distribution of different numerical variables on top of each other


# In[31]:


#First we convert the categorical target variable outcomes into numerical for easy numerical analllysis:
telecom_base_data['Churn'] = np.where(telecom_base_data.Churn == 'Yes',1,0)
#Next we convert categorical variables into numerical by getting dummy variables this makes it easy for regression and corelation analysis
telecom_base_dummies=pd.get_dummies(telecom_base_data)
telecom_base_dummies.head()


# In[32]:


#Scatter plot between monthly charges and Total Charges
sns.lmplot(data=telecom_base_data,x='MonthlyCharges',y='TotalCharges',fit_reg=True)
#We can see as monthly charges increase total charges also increase


# In[33]:


#Now we see churn by monthly charges and total charges using k-density plot


# In[34]:


#For monthly charges
Mth = sns.kdeplot(telecom_base_dummies.MonthlyCharges[(telecom_base_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(telecom_base_dummies.MonthlyCharges[(telecom_base_dummies["Churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')

#As monthly charges increase churn rate also increases


# In[35]:


#for total charges
Tot = sns.kdeplot(telecom_base_dummies.TotalCharges[(telecom_base_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Tot = sns.kdeplot(telecom_base_dummies.TotalCharges[(telecom_base_dummies["Churn"] == 1) ],
                ax =Tot, color="Blue", shade= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total charges by churn')

#As total charges increase the churn rate is low, also we must remember as tenure increases total charge increases


# In[36]:


telecom_base_dummies.corr()


# In[37]:


#To find out correlation between different predictor variables on target variable churn , we plot a corelation bar graph to see the effect of each variable:
plt.figure(figsize=(20,8))
telecom_base_dummies.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')


# In[38]:


#Following insights can be gained from the corelation plot:
#1-month to month contracct,no online service,no tech support ,one year tenure group,fibre_optic internet service,electric_cheque payment method
#no device protection,paperless billing,no_dependents etc are the main features of churners


# In[39]:


#2-On the other hand two-year contract,device protection,online backup,tech_support, streaming_movies without internet service ,tenure group of 5plus years,etc are the features of non churners


# In[40]:


#3-Gender, phone service,multiple lines etc seem to have no effect on customer churn


# In[41]:


#Bi-variate analysis
# Specify the variable pairs for histogram plotting
new_df1_target0=telecom_data.loc[telecom_data["Churn"]==0]
new_df1_target1=telecom_data.loc[telecom_data
                                 ["Churn"]==1]


# In[42]:


def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright') 
        
    plt.show()


# In[43]:


telecom_base_data.head()


# In[44]:


uniplot(new_df1_target1,col='Partner',title='Distribution of Gender for Churned Customers',hue='gender')


# In[45]:



uniplot(new_df1_target1,col='Contract',title='Distribution of Contract for Churned Customers',hue='tenure_group')


# In[46]:




uniplot(new_df1_target1,col='Contract',title='Distribution of Contract for Churned Customers',hue='tenure_group')


# In[47]:


uniplot(new_df1_target1,col='StreamingTV',title='Distribution of Customers StreamingTV and StreamingMovies for Churned Customers',hue='StreamingMovies')


# In[48]:


uniplot(new_df1_target1,col='DeviceProtection',title='Distribution of Customers DeviceProtection and TechSupport for Churned Customers',hue='TechSupport')


# In[49]:


uniplot(new_df1_target1,col='OnlineSecurity',title='Distribution of Customers with Online Security and OnlineBackup for Churned Customers',hue='OnlineBackup')


# In[50]:


uniplot(new_df1_target1,col='PaymentMethod',title='Distribution of Customers based on Payment Method and PaperlessBilling for Churned Customers',hue='PaperlessBilling')


# In[51]:


uniplot(new_df1_target1,col='PaymentMethod',title='Distribution of Customers based on Payment Method and PaperlessBilling for Churned Customers',hue='PaperlessBilling')


# In[52]:


uniplot(new_df1_target1,col='PaymentMethod',title='Distribution of Customers based on Payment Method and PaperlessBilling for Churned Customers',hue='PaperlessBilling')


# In[53]:


telecom_base_dummies


# In[54]:


#Building Logistic regression model to predict likelihood of churning 
#First we need to do upsampling as data set is imbalanced
x=telecom_base_dummies.drop('Churn',axis=1)
y=telecom_base_dummies['Churn']


# In[55]:


x
y


# In[56]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[57]:


get_ipython().system('pip install imbalanced-learn')


# In[58]:


from imblearn.combine import SMOTEENN

sm = SMOTEENN()
x_resampled, y_resampled = sm.fit_resample(x, y)


# In[59]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Drop the 'Churn' column and use the remaining columns as predictors
X = telecom_base_dummies.drop('Churn', axis=1)
y = telecom_base_dummies['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit logistic regression model
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)

# Predictions
y_pred_train = log_reg_model.predict(X_train)
y_pred_test = log_reg_model.predict(X_test)

# Evaluate model
print("Training Classification Report:")
print(classification_report(y_train, y_pred_train))
print("Testing Classification Report:")
print(classification_report(y_test, y_pred_test))

# Confusion Matrix
print("Confusion Matrix for Training Data:")
print(confusion_matrix(y_train, y_pred_train))
print("Confusion Matrix for Testing Data:")
print(confusion_matrix(y_test, y_pred_test))


# In[108]:





# In[ ]:





# In[102]:


# Predict probabilities for churners using the trained logistic regression model
telecom_base_data['Probability_Churn'] = log_reg_model.predict_proba(X)[:, 1]

# Calculate probabilities for non-churners (1 - Probability_Churn)
telecom_base_data['Probability_No_Churn'] = 1 - telecom_base_data['Probability_Churn']



# In[103]:


telecom_base_data.head()


# In[109]:


customer_id=pd.read_csv("C:\\Users\\VATTI VAMSHI\\Downloads\\WA_Fn-UseC_-Telco-Customer-Churn (1).csv")


# In[124]:


customer_id["customerID"]


# In[125]:


telecom_base_data["customerID"]=customer_id["customerID"]


# In[126]:


telecom_base_data.head()


# In[127]:


telecom_base_data.to_csv('telecom_predicted.csv',index=False)


# In[ ]:




