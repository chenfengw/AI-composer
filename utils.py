#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)
import math


def generate_dictionary(music_files):
    '''
    music_files {list} -- list of strings that contrain filenames of 
    each file. ie ['train.txt','val.txt'...]
    '''
    assert isinstance(music_files,list)
    counter = 3
    dictionary = {"<start>" : 0 , "<end>" : 1, "\n" : 2}
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



#character_dictionary = generate_dictionary('train.txt')


def read_song(filename):
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

#song_dictionary = read_song('train.txt')

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
    encoded_array = np.zeros((1,size_song,95))
    counter = 0 
    char_index = 0
    
    # Generate the one hot encoded matrix
    while True:
        if char_index == 0:
            char = "<start>"
        elif char_index == (len(song) - 5):
            char = "<end>"
        else:
            char = song[char_index]
        position = character_dictionary[char]
        encoded_array[0][counter][position] = 1
        counter += 1
        if char_index == 0:
            char_index += 7
        elif char_index == (len(song)-5):
            break
        else:
            char_index += 1
            
    # Selectively choose the chunk of the one hot encoding we want and 
    #append it to a running list
    start_idx = 0
    end_idx = 100
    result = np.zeros([end_idx,len(character_dictionary)])
    
    while end_idx <= (encoded_array.shape[1] + 100):
        if end_idx <= (encoded_array.shape[1]):
            input_list.append(encoded_array[0][start_idx : end_idx][:])
        else:
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
            extract = encoded_array[0][start_idx : end_idx - 1][:]
            result[:extract.shape[0],:extract.shape[1]] = extract
            target_list.append(result)
            
        start_idx += 100
        end_idx += 100
    
    return input_list, target_list 



