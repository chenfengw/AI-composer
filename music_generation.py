#!/usr/bin/env python
# coding: utf-8

# In[14]:


from utils import *


# In[20]:


################################# Hypyer parameter #################################
input_dim = 96    # length of one hot encoded character --> 95
hidden_dim = 75  # number of hidden neuron
n_layers = 1      # number of hidden layers
batch_size = 1
model_path = 'hidden75.pt'
T = 0.7
weighted_sample = True
n_loop = 10000
hidden_index = 0
np.random.seed(42)


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
    rnn.load_state_dict(torch.load(model_path))
    print("Previous model state loaded successfully")
else:
    print("Model not loaded. No previous model found.")

rnn = rnn.to(computing_device)


# In[18]:


############# load data ###############
character_dic = generate_dictionary(['train.txt','val.txt','test.txt'])
rev_dic = {v:k for k,v in character_dic.items()}
rev_dic[character_dic['\n']] = "nl"
rev_dic[character_dic[' ']] = "sp"
test_song_dic = read_song('test.txt')


# In[52]:


############# compute test loss ############
test_loss_chunks = []
with torch.no_grad():
    for song_index in range(len(test_song_dic)):
        hidden_state = torch.zeros(n_layers, batch_size, hidden_dim).to(computing_device)
        cell_state = torch.zeros(n_layers, batch_size, hidden_dim).to(computing_device)
        hidden = (hidden_state, cell_state)

        # draw 1 song at a time
        song_raw = test_song_dic[song_index]

        # convert song to one-hot
        input_lists, target_lists = encode_song(song_raw, character_dic)

        for chunk_index, (input_list, target_list) in enumerate(zip(input_lists,target_lists)):
            # convert input, and target to tensor
            input_tensor = torch.FloatTensor(input_list).unsqueeze_(0)
            target_tensor = torch.from_numpy(target_list.argmax(1))

            # send all data to GUP
            input_tensor, target_tensor= input_tensor.to(computing_device), target_tensor.to(computing_device)

            # Run our forward pass.
            output, hidden = rnn(input_tensor,hidden)

            # compute loss, run optimizer step
            loss = loss_function(output, target_tensor)

            # save validation loss
            test_loss_chunks.append(loss.item())

print("test loss: {}".format(np.array(test_loss_chunks).mean()))


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
print(decode_song(char_index_list,character_dic))


# In[ ]:


# need to reshape data in to rectangle
data = test
heatmap = plt.pcolor(data)

for x in range(data.shape[1]):
    for y in range(data.shape[0]):
        plt.text(y + 0.5, x + 0.5, "%s" %rev_dic[char_index_array[x,y]],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )

plt.colorbar(heatmap)
plt.savefig("heatmap",dpi=600)
plt.show()

