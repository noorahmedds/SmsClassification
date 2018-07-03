
# coding: utf-8

# In[3]:


# Spam or ham
# We have to train our model on spam and ham
# The model will be a naive bayes classifier
import numpy as np
import pandas as pd


# In[19]:





# In[49]:


# filePath = "/Users/noorahmed/Desktop/untitled folder/spam.csv"
df = pd.read_csv("spam2.csv", encoding='latin-1')


# In[57]:


df.columns


# In[58]:


df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace=True)


# In[65]:


df.shape[0]


# In[92]:


# Divide the dataframe into train and test
from sklearn.model_selection import train_test_split


# In[93]:


X_train, X_test, y_train, y_test = train_test_split(df['v2'].values, df['v1'].values, test_size=0.33, random_state=42)


# In[114]:


train_df = pd.DataFrame({'v1':y_train, 'v2':X_train})
train_df.head()


# In[116]:


test_df = pd.DataFrame({'v1':y_test, 'v2':X_test})
test_df.head()


# In[139]:


# Lets determine some of the variables for our naive bayes classifier

# For m estimator
p = 0
m = 0
n = []

# Prior probabilities (ham, spam)
Ph = 0
Ps = 0


# In[141]:




def getPrior(data, hypothesis = ""):
    prior = 0
    try:
        if hypothesis == "": 
            raise Exception();
        
        prior_count = 0
        for row in range(0, data.shape[0]):
            if hypothesis == data['v1'].iloc[row]:
                prior_count += 1
        prior = prior_count / data.shape[0]
        
        return prior
            
        
    except Exception as inst:
        print(inst)
        print("hypothesis passed cannot be empty")
        

print(getPrior(train_df, "ham") + getPrior(train_df, "spam"))
Ph = getPrior(train_df, "ham")
Ps = getPrior(train_df, "spam")

n.append(Ph*(train_df.shape[0]))
n.append(Ps*(train_df.shape[0]))


# In[142]:


vocab = set()
vocab.clear()


# In[143]:


# Now we train to determine our Evidence probabilities
# So P(ai|vj)



def vocabularyCount(data):
    for row in range(0, data.shape[0]): 
        message = data['v2'].iloc[row]
        for word in message:
            vocab.add(word)
            
            
    return len(vocab)

m = vocabularyCount(train_df)
p = 1/m


# In[91]:


# No preprocessing done here. Should apply some preprocessing


# In[126]:


words_count = {
    'ham':{},
    'spam':{}
}
    


# In[137]:


def train(data):
    for row in range(0, data.shape[0]): 
        message = data['v2'].iloc[row]
        msg = message.split()
        for word in msg:
            hypothesis = data['v1'].iloc[row]
            if word not in words_count[hypothesis].keys():
                words_count[hypothesis][word] = 0
            else:
                words_count[hypothesis][word] += 1
            

train(train_df)


# In[138]:


words_count
# words count now contains the count for all words in each category/label/hypothesis whatever you wanna call it
# we should probably remove stop words but for this preliminary analysis i wont


# In[144]:


# here we simply just divide all words by the probability of the hypothesis occuring
for outer_key in words_count.keys():
    for inner_key in words_count[outer_key].keys():
        n_ind = 0
        if outer_key == "ham": 
            n_ind = 0
        else:
            n_ind = 1
        words_count[outer_key][inner_key] = words_count[outer_key][inner_key] / n[n_ind]
        


# In[ ]:


# Create m estimator function
# Create accuracy function

