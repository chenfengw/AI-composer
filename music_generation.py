#!/usr/bin/env python
# coding: utf-8

# In[14]:

import torch
from utils import *


# In[20]:


################################# Hypyer parameter #################################
input_dim = 96    # length of one hot encoded character --> 96
hidden_dim = 75  # number of hidden neuron
n_layers = 1      # number of hidden layers
batch_size = 1
model_path = 'model_pretrain.pt'
T = 0.7
weighted_sample = True
n_loop = 10000
hidden_index = 0
np.random.seed(42)
torch.manual_seed(43)
music_output = "music_generated.txt"

# In[16]:


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


# In[17]:


##################### Load network ##########################
rnn = RNN(input_dim, hidden_dim, n_layers, batch_size)
loss_function = nn.CrossEntropyLoss()

if os.path.exists(model_path):
    print("Loading previous model state")
    rnn.load_state_dict(torch.load(model_path,map_location=computing_device))
    print("Previous model state loaded successfully")
else:
    print("Model not loaded. No previous model found.")

rnn = rnn.to(computing_device)


# In[18]:
############# load data ###############
character_dic = generate_dictionary(['data/train.txt','data/val.txt','data/test.txt'])
rev_dic = {v:k for k,v in character_dic.items()}
rev_dic[character_dic['\n']] = "nl"
rev_dic[character_dic[' ']] = "sp"

# In[23]:
############# generate music  ############
char_index_list = []
char_index_list.append(1)
hidden_list = []
count = 0

with torch.no_grad():
    char_index = 1
    input_tensor = convert_onehot_tensor(char_index,character_dic)
    hidden_state = torch.zeros(n_layers, batch_size, hidden_dim).to(computing_device)
    cell_state = torch.zeros(n_layers, batch_size, hidden_dim).to(computing_device)
    hidden = (hidden_state, cell_state)
    
    while char_index != 2 and count<n_loop:
        # send all data to GUP
        input_tensor = input_tensor.to(computing_device)

        # Run our forward pass.
        output, hidden = rnn(input_tensor,hidden)
        
        # collect the hidden state
        hidden_value = hidden[0][0,0,hidden_index]
        hidden_list.append(hidden_value.item())
        
        # put output through softmax
        output_softmax = softmax(output,T)
        
        # weighted sampling
        if weighted_sample:
            char_index = list(sampler.WeightedRandomSampler(output_softmax, 1, replacement=True))[0][0]
        else:
            char_index = output_softmax.argmax().item()
        
        # collect the char index 
        char_index_list.append(char_index)
        
        # feed output into input of next time step
        input_tensor = convert_onehot_tensor(char_index,character_dic)
    
        count += 1

# save generated music
music = decode_song(char_index_list,character_dic)
with open(music_output,"w") as f:
    f.write(music)
print("music saved to: {}".format(music_output))
