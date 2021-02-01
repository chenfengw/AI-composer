#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *


# In[2]:


################################# Hypyer parameter #################################
alpha = 0.0001
epoch = 30
input_dim = 96    # length of one hot encoded character --> 96
hidden_dim = 75  # number of hidden neuron
n_layers = 1      # number of hidden layers
batch_size = 1
early_stop = True
early_stop_epoch = 3
np.random.seed(42)
model_name = "model.pt"

# In[3]:


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


# In[4]:


############# load data ###############
character_dic = generate_dictionary(['data/train.txt','data/val.txt','data/test.txt'])
train_song_dic = read_song('data/train.txt')
valid_song_dic = read_song('data/val.txt')
test_song_dic = read_song('data/test.txt')


# In[5]:


rnn = RNN(input_dim, hidden_dim, n_layers, batch_size).to(computing_device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(),lr = alpha)

train_loss = []
valid_loss = []
valid_increase_count = 0

for i in range(epoch):
    train_loss_epoch = []
    valid_loss_epoch = []
    
    # training model
    train_rand_index = np.arange(len(train_song_dic))
    np.random.shuffle(train_rand_index)
    for song_index, song in enumerate(train_rand_index):
        hidden_state = torch.zeros(n_layers, batch_size, hidden_dim).to(computing_device)
        cell_state = torch.zeros(n_layers, batch_size, hidden_dim).to(computing_device)
        hidden = (hidden_state, cell_state)
        
        # draw 1 song at a time
        song_raw = train_song_dic[song]
        
        # convert song to one-hot
        input_lists, target_lists = encode_song(song_raw, character_dic)
        
        for chunk_index, (input_list, target_list) in enumerate(zip(input_lists,target_lists)):
            # clear gradient
            optimizer.zero_grad()
            
            # convert input, and target to tensor
            input_tensor = torch.FloatTensor(input_list).unsqueeze_(0)
            target_tensor = torch.from_numpy(target_list.argmax(1))
            
            # send all data to GUP
            input_tensor, target_tensor= input_tensor.to(computing_device), target_tensor.to(computing_device)
            
            # Run our forward pass.
            output, hidden = rnn(input_tensor,hidden)
            
            # compute loss, run optimizer step
            loss = loss_function(output, target_tensor)
            loss.backward()
            optimizer.step()
            
            # detech hidden state
            hidden = (hidden[0].detach(), hidden[1].detach())
            
            # print status of training
            print('Epoch #{}, Train song #{}, #chunk #{}, loss={}'.format(i+1,song_index,chunk_index,loss.item()))

            # save training loss
            train_loss_epoch.append(loss.item())
            
    train_loss.append(np.array(train_loss_epoch).mean())
    
    # compute validation loss
    with torch.no_grad():
        for song_index in range(len(valid_song_dic)):
            hidden_state = torch.zeros(n_layers, batch_size, hidden_dim).to(computing_device)
            cell_state = torch.zeros(n_layers, batch_size, hidden_dim).to(computing_device)
            hidden = (hidden_state, cell_state)

            # draw 1 song at a time
            song_raw = valid_song_dic[song_index]

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
                
                # print valid status
                print('Epoch #{}, Valid song #{}, #chunk #{}, loss={}'.format(i+1,song_index,chunk_index,loss.item()))
                
                # save validation loss
                valid_loss_epoch.append(loss.item())

        valid_loss.append(np.array(valid_loss_epoch).mean())
    
    # implement early stoping
    if early_stop:
        # count number of increase in valid loss
        if len(valid_loss) > 1 and (valid_loss[-1] > valid_loss[-2]):
            valid_increase_count += 1
            
        # if increase consecutively for early_stop_epoch, break
        if valid_increase_count >= early_stop_epoch:
            print("early stop trigered, stop at {} epoch".format(i+1))
            break
        
    # save the model after 1 epoch
    torch.save(rnn.state_dict(), model_name)
    print("model: {} saved!!".format(model_name))

# In[9]:


plt.figure()
plt.plot(np.arange(len(train_loss))+1,train_loss,label="traning loss")
plt.plot(np.arange(len(valid_loss))+1,valid_loss,label="valid loss")
plt.title("Train/Vlid Loss")
plt.legend()
plt.show()




