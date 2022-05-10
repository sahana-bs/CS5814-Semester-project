#!/usr/bin/env python
# coding: utf-8

# In[3]:


#dependency according to https://github.com/salesforce/CodeT5#dependency
get_ipython().system('pip install transformers==4.6.1')


# In[4]:


#imports
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


# In[5]:


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


# In[7]:


from google.colab import drive
drive.mount('/content/gdrive/')
data_path = '/content/gdrive/My Drive/CS5814/Project'


# In[8]:


import os
os.chdir(data_path+'/javaCorpus/token_completion')


# In[9]:


#preprocess the data
get_ipython().system('python preprocess_java.py --base_dir=token_completion --output_dir=token_completion')


# In[10]:


get_ipython().system('ls')


# In[11]:


class Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.code_
        self.ctext = self.data.code_

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        # source = self.tokenizer.batch_encode_plus([self.text], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        # target = self.tokenizer.batch_encode_plus([self.otext], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')
        # print([self.text][0])
        source = self.tokenizer([text], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer([ctext], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


# In[12]:


def train(epoch, tokenizer, model, device, loader, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loss = []
    torch.save(model,'my_checkpoint.pth.tar')
    torch.save(model, data_path+'my_checkpoint.pth.tar')
    model.train()
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]
        train_loss.append(loss.item())
        if _%100==0:
          print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("=> Saving checkpoint")        
    return train_loss


# In[13]:


def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=550, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


# In[14]:


Train_batch  = 8
Valid_batch = 8
epochs = 1
val_epochs = 1
learning_rate = 1e-4
seed = 42
max_len = 512
output_len = 150


# In[15]:


# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(seed) # pytorch random seed
np.random.seed(seed) # numpy random seed
torch.backends.cudnn.deterministic = True

# tokenzier for encoding the text
tokenizer = RobertaTokenizer.from_pretrained("microsoft/CodeGPT-small-java-adaptedGPT2")

train_data = pd.read_csv('train.txt', sep='<s>', header = None)
dev_data = pd.read_csv('dev.txt', sep='<s>', header = None)

#apply <s> at the beginning
train_data.columns = ['idx', 'code']
train_data['code_'] = train_data['code'].apply(lambda x:'<s> '+x)

dev_data.columns = ['idx', 'code']
dev_data['code_'] = dev_data['code'].apply(lambda x:'<s> '+x)

print("TRAIN Dataset: {}".format(train_data.shape))
print("VAL Dataset: {}".format(dev_data.shape))

#calling the dataset class 
training_set = Dataset(train_data, tokenizer, max_len, output_len)
val_set = Dataset(dev_data, tokenizer, max_len, output_len)

# Defining the parameters for creation of dataloaders
train_params = {
    'batch_size': Train_batch,
    'shuffle': True,
    'num_workers': 2
    }

val_params = {
    'batch_size': Valid_batch,
    'shuffle': False,
    'num_workers': 2
    }

# Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
training_loader = DataLoader(training_set, **train_params)
val_loader = DataLoader(val_set, **val_params)

model = T5ForConditionalGeneration.from_pretrained('microsoft/CodeGPT-small-java-adaptedGPT2')
model = model.to(device)

# Defining the optimizer that will be used to tune the weights of the network in the training session. 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[16]:


# Training loop
for epoch in range(epochs):
    training_loss = train(epoch, tokenizer, model, device, training_loader, optimizer)


# In[ ]:


## Training accuracy vs number of epochs
# training_loss = [11.594493865966797, 0.14940930902957916, 0.09707176685333252, 0.0923515036702156, 0.05420288071036339, 0.0764363557100296,0.04925357550382614,0.058485135436058044, 0.08185863494873047, 0.031726688146591187]
plt.plot(training_loss)
plt.xlabel('Iterations')
plt.ylabel('Training loss')


# In[17]:


test = pd.read_json('test.json', lines=True)


# In[18]:


test


# In[19]:


test.columns
test.rename(columns={'input': 'code_'}, inplace=True)


# In[20]:


test['gt']


# In[ ]:


#-------------------------------------------------------------------
# Testing and saving the results to a dataframe
#-------------------------------------------------------------------

val_epochs = 1
test_set = Dataset(test, tokenizer, max_len, output_len)
test_params = {
    'batch_size': 8,
    'shuffle': False,
    'num_workers': 2
    }
test_loader = DataLoader(test_set, **val_params)
for epoch in range(val_epochs):
    predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})


# In[ ]:


#---------------------------------------------------------------
#result metrics
#---------------------------------------------------------------
from nltk.metrics import edit_distance    
final_df["distance"] = final_df.loc[:, ["Generated Text","Actual Text"]].apply(lambda x: edit_distance(*x), axis=1)


# In[ ]:


final_df["exact_match"] = final_df.loc[:, ["Generated Text","Actual Text"]].apply(lambda x: 1 (if x["Generated Text"].split()==x["Actual Text"].split()) else 0)


# In[ ]:


print("Edit distance: ", final_df["distance"].mean())


# In[ ]:


final_df


# In[ ]:


final_df.to_csv('predictions.csv')


# In[ ]:


model = torch.load(data_path+'my_checkpoint.pth.tar')


# In[ ]:


#Example inference

text = "def (user): print(f'hello <extra_id_0>!')"
input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate one code span
generated_ids = model.generate(input_ids, max_length = 50)   #line_generation
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))


# # Token level inferencing

# In[ ]:


#getting the test file for token completion task

get_ipython().system('wget -O test.txt https://zenodo.org/record/3628665/files/java_test_pre')


# In[ ]:


#--------------------------------------------------------------------------------
#token level inferencing
#--------------------------------------------------------------------------------
test_token = pd.read_csv("test.txt", sep='<s>', header = None)
test_token.columns = ['idx', 'code']
test_token['code_'] = test_token['code'].apply(lambda x:'<s> '+x)


# In[ ]:


test_token


# In[ ]:


tokenizer = RobertaTokenizer.from_pretrained('microsoft/CodeGPT-small-java-adaptedGPT2')

test_token_set = Dataset(test_token, tokenizer, max_len, output_len)
test_params = {
    'batch_size': 8,
    'shuffle': False,
    'num_workers': 2
    }
test_loader = DataLoader(test_token_set, **test_params)
for epoch in range(val_epochs):
    predictions, actuals = validate(epoch, tokenizer, model, device, test_loader)


# In[ ]:


tmp = pd.read_csv("predictions_token.csv", index_col=0)


# In[ ]:


predictions = list(tmp['Generated Text'])
actuals = list(tmp['Actual Text'])


# In[ ]:


#-------------------------------------------------------------
#accuracy of predictions..
#Based on the evaluator.py script from CodeXGLUE
#--------------------------------------------------------------

total = 0
correct = 0.0
for pred, gt in zip(predictions, actuals):
    pred = pred.split()
    gt = gt.split()
    for x, y in zip(pred, gt):
        if y not in ["<s>", "</s>", "<EOL>", "<pad>"]:
            total += 1
            if x == y:
                correct += 1
print((f"Total {total} tokens, accuracy: {round(correct/total*100, 2)}"))


# In[ ]:


text = "<s> import json <EOL> json . load ( f ) </s>"
input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate one code span
generated_ids = model.generate(input_ids, max_length = 10)   #token completion example
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))


# In[ ]:


#------------------------------------------------------
#understanding the data (should've been done at the beginning)
#------------------------------------------------------

data = pd.read_csv("train.csv", sep='<s>', header = None)


# In[ ]:


data['CodeLength'] = data['code'].apply(lambda x:len(x.split()))


# In[ ]:


#check the highest, lowest and the mean of the lengths of the code
print("Maximum code length ", max(data['CodeLength']))
print("Minimum code length ", min(data['CodeLength']))
print("mean code length ", data['CodeLength'].mean())


# # Converting to python file

# In[29]:


os.chdir(data_path)
get_ipython().system('ls')


# In[30]:


get_ipython().system('jupyter nbconvert --to python Final_Project_DL_CodeGPT.ipynb')

