import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
from utils import *


def train(model, loss, optimizer, data, computing_device, epoch=1, model_name="model.pt"):
    """Training model

    Args:
        model (nn.LSTM): LSTM model
        loss (nn.CrossEntorpy): loss function
        optimizer (nn.optim): SGD or Adam
        data (dict): song dictionary.
        computing_device (torch.device): device for training
        epoch (int, optional): epoch to train. Defaults to 1.
        model_name (str, optional): model name to save. Defaults to "model.pt".

    Returns:
        list: training loss per epoch
    """
    train_loss = []
    for i in range(epoch):
        train_loss_epoch = []

        # randomize train song index
        train_rand_index = np.arange(len(train_song_dic))
        np.random.shuffle(train_rand_index)

        # one song at a time
        for song_index, song in enumerate(train_rand_index):
            # hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
            # cell_state = torch.randn(n_layers, batch_size, hidden_dim)
            hidden_state = torch.zeros(n_layers, batch_size, hidden_dim)
            cell_state = torch.zeros(n_layers, batch_size, hidden_dim)
            hidden = (hidden_state, cell_state)

            # each 1 song at a time
            song_raw = train_song_dic[song]

            # convert song to one-hot
            input_lists, target_lists = encode_song(song_raw, character_dic)

            # feed chunks of a song
            for chunk_index, (input_list, target_list) in enumerate(zip(input_lists, target_lists)):
                # We need to clear them out before each instance
                # optimizer.zero_grad()

                # convert input, and target to tensor
                input_tensor = torch.FloatTensor(input_list).unsqueeze_(0)
                target_tensor = torch.from_numpy(target_list.argmax(1))

                # send all data to GUP
                input_tensor, target_tensor = input_tensor.to(
                    computing_device), target_tensor.to(computing_device)

                # Run our forward pass.
                output, hidden = model(input_tensor, hidden)

                # compute loss, run optimizer step
                loss = loss_function(torch.squeeze(output), target_tensor)
                loss.backward()
                optimizer.step()

                # detach hidden state
                hidden = (hidden[0].detach(), hidden[1].detach())

                # print status of training
                print('Train song #{}, #chunk #{}, loss={}'.format(song_index,
                                                                   chunk_index,
                                                                   loss.item()))
                # save training loss
                train_loss_epoch.append(loss.item())

        # save loss per epoch
        train_loss.append(np.array(train_loss_epoch).mean())

    # save model
    torch.save(model.state_dict(), model_name)
    print("{} saved!!".format(model_name))
    return train_loss


if __name__ == "__main__":
    # Hypyer parameter
    alpha = 0.001
    epoch = 1
    input_dim = 95    # length of one hot encoded character --> 95
    hidden_dim = 95  # number of hidden neuron
    n_layers = 1      # number of hidden layers
    batch_size = 1
    seq_len = 100
    np.random.seed(42)
    model_name = "model.pt"

    # Setup GPU optimization if CUDA is supported
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 1, "pin_memory": True}
        print("CUDA is supported")
    else:  # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    # load data
    character_dic = generate_dictionary(['train.txt', 'val.txt', 'test.txt'])
    train_song_dic = read_song('train.txt')
    valid_song_dic = read_song('val.txt')
    test_song_dic = read_song('test.txt')

    # initialization
    model = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True).to(
        computing_device)  # input models
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=alpha)

    # train
    loss_train = train(model, loss_function, optimizer,
                       train_song_dic, computing_device, 
                       epoch=epoch, model_name=model_name)
