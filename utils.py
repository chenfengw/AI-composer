#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.sampler as sampler 
import numpy as np
import sys
import os
from matplotlib import pyplot as plt

def softmax(input_tensor,T):
    '''
    softmax function with Temperature
    '''
    data_exp = torch.exp(input_tensor/T)
    return data_exp/data_exp.sum()

def convert_onehot_tensor(char_index,character_dic):
    '''
    Convert char index to one hot encoded tensor of right dimension
    '''
    input_tensor = torch.zeros(1,len(character_dic)) 
    input_tensor[:,char_index] = 1
    input_tensor = input_tensor.unsqueeze_(0)
    
    return input_tensor

class RNN(nn.Module):
    '''RNN class that incorporate LSTM and linear layer  
    
    Extends:
        nn.Module
    '''
    def __init__(self, input_dim, hidden_dim, n_layers, batch_size):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        return self.fc(lstm_out), hidden
    
    def init_hidden(self):
        # weight = next(self.parameters()).data
        # hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
        #               weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

        # return hidden
        hidden_state = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(computing_device)
        cell_state = torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).to(computing_device)
        hidden = (hidden_state, cell_state)

        return hidden



def generate_dictionary(music_files):
    '''
    music_files {list} -- list of strings that contrain filenames of 
    each file. ie ['train.txt','val.txt'...]
    '''
    assert isinstance(music_files,list)
    counter = 4
    dictionary = {"%":0,"<start>" : 1 , "<end>" : 2, "\n" : 3}
    for file in music_files:
        with open(file) as f:
            line = f.readline()
            while line: 
                if "<start>" not in line or "<end>" not in line:
                    for letter in line:
                        if letter not in dictionary:
                            dictionary[letter] = counter
                            counter +=1
                line = f.readline()
     
    return dictionary

def read_song(filename):
    '''Read songfile.txt and create a dictionary that saves all the song.
    keys are index of song, and values are content of that song.
    
    Arguments:
        filename {str} -- file name. ie. "train.txt"
    
    Returns:
        dic -- dictionary that contains all the songs. 
        keys are index of the song, and values are content of that song.
    '''
    song_dictionary = {}
    counter = 0
    with open(filename) as f:
        line = f.readline()
        while line:
            if counter not in song_dictionary:
                song_dictionary[counter] = line
            else:
                song_dictionary[counter] = song_dictionary[counter] + line 
            if "<end>" in line:
                counter +=1
            line = f.readline()
    return song_dictionary


def encode_song(song , character_dictionary):
    """
    converts song to list of one-hot encoded numpy arrays
    
    song : string representation of song
    character_dictionary : every unique character mapped to an index
    
    return : list of 100 x 95 x 1 numpy array
    """
    size_song = (len(song) - 10)
    #print(size_song)
    input_list = []
    target_list = []
    encoded_array = np.zeros((1,size_song,96))
    counter = 0 
    char_index = 0
    
    # Generate the one hot encoded matrix
    while True:
        if char_index == 0:
            char = "<start>"
        elif char_index == (len(song) - 6):
            char = "<end>"
        else:
            char = song[char_index]
        position = character_dictionary[char]
        encoded_array[0][counter][position] = 1
        counter += 1
        if char_index == 0:
            char_index += 7
        elif char_index == (len(song)-6):
            break
        else:
            char_index += 1
            
    # Selectively choose the chunk of the one hot encoding we want and 
    #append it to a running list
    start_idx = 0
    end_idx = 100
    
    
    while end_idx <= (encoded_array.shape[1] + 100):
        if end_idx <= (encoded_array.shape[1]):
            input_list.append(encoded_array[0][start_idx : end_idx][:])
        else:
            result = np.zeros([100,len(character_dictionary)])
            extract = encoded_array[0][start_idx : counter - 1][:]
            result[:extract.shape[0],:extract.shape[1]] = extract
            input_list.append(result)
        start_idx += 100
        end_idx += 100
    
    start_idx = 1
    end_idx = 101
    while start_idx < (encoded_array.shape[1]):
        if ( end_idx < encoded_array.shape[1]):
            target_list.append(encoded_array[0][start_idx : end_idx][:])
        else:
            result = np.zeros([100,len(character_dictionary)])
            extract = encoded_array[0][start_idx : end_idx - 1][:]
            result[:extract.shape[0],:extract.shape[1]] = extract
            target_list.append(result)
            
        start_idx += 100
        end_idx += 100
    
    return input_list, target_list 


def decode_song(char_index_list,dic):
    '''
    Decode the song
    
    char_list -- list
    dic -- original char dic
    '''
    # flip the dictionary first
    inv_map = {v: k for k, v in dic.items()}
    
    # devode the list using the maping in dictionary
    str_list = [inv_map[i] for i in char_index_list]
    str_music = "".join(str_list)
    
    return str_music
    