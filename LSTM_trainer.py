# In[31]:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
from utils import *

# In[32]:

################################# Hypyer parameter #################################
alpha = 0.001
epoch = 2
input_dim = 95    # length of one hot encoded character --> 95
hidden_dim = 95  # number of hidden neuron
n_layers = 1      # number of hidden layers
batch_size = 1
seq_len = 100
np.random.seed(42)


# In[33]:


# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")


# In[34]:


############# locad data ###############
character_dic = generate_dictionary(['train.txt','val.txt','test.txt'])
train_song_dic = read_song('train.txt')
valid_song_dic = read_song('val.txt')
test_song_dic = read_song('test.txt')

# In[18]:
train_rand_index = np.arange(len(train_song_dic))
np.random.shuffle(train_rand_index)

# In[35]:
model = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True).to(computing_device) # input models
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = alpha)

train_loss = []
valid_loss = []

for i in range(epoch):
    train_loss_epoch = []
    valid_loss_epoch = []
    
    # randomize train song index
    train_rand_index = np.arange(len(train_song_dic))
    np.random.shuffle(train_rand_index)
    
    for song_index, song in enumerate(train_rand_index):
#         hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
#         cell_state = torch.randn(n_layers, batch_size, hidden_dim)
        hidden_state = torch.zeros(n_layers, batch_size, hidden_dim)
        cell_state = torch.zeros(n_layers, batch_size, hidden_dim)
        hidden = (hidden_state, cell_state)
        
        # each 1 song at a time
        song_raw = train_song_dic[song]
        
        # convert song to one-hot
        input_lists, target_lists = encode_song(song_raw, character_dic)
        
        for chunk_index, (input_list, target_list) in enumerate(zip(input_lists,target_lists)):
            # We need to clear them out before each instance
            #optimizer.zero_grad()
            
            # convert input, and target to tensor
            input_tensor = torch.FloatTensor(input_list).unsqueeze_(0)
            target_tensor = torch.from_numpy(target_list.argmax(1))
            
            # send all data to GUP
            input_tensor, target_tensor= input_tensor.to(computing_device), target_tensor.to(computing_device)

            # Run our forward pass.
            output, hidden = model(input_tensor,hidden)

            # compute loss, run optimizer step
            loss = loss_function(torch.squeeze(output), target_tensor)
            loss.backward()
            optimizer.step()
            
            # detech hidden state
            hidden = (hidden[0].detach(), hidden[1].detach())
            
            # print status of training
            print('Train song #{}, #chunk #{}, loss={}'.format(song_index,chunk_index,loss.item()))

            # save training loss
            train_loss_epoch.append(loss.item())
  
    train_loss.append(np.array(train_loss_epoch).mean())


# %% save model
torch.save(model.state_dict(), PATH)