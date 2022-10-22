import os
import torch
from torch.utils.data import Dataset
import random
import argparse
import json

class Triplets(Dataset):
    '''
    this is a custom dataset class, containing triplets
    of a given knowledge graph. Should be in appropriate form...
    '''

    def __init__(self, path = None,
        unique_objects = None, unique_relationships = None):

        self.unique_objects = dict() if not unique_objects else unique_objects
        self.unique_relationships = dict() if not unique_relationships else unique_relationships
        self.triplets = list()

        n_objects = 0
        n_relationships = 0

        #path of a knowledge graph!
        if path:
            with open(path, 'r') as file:
                for line in file:
                    #tab separated values!!!
                    h, l, t = line.split('\t')
                    #remove \n
                    t = t[:-1]

                    #we will encode the nodes and edges with unique integers!
                    #this will match with the embedding Tensors that are used to contain
                    #embeddings for our objects!
                    if h not in self.unique_objects:
                        #if unique_objects exists, throw that triple!
                        if unique_objects: continue
                        self.unique_objects[h] = n_objects
                        n_objects += 1
                    if t not in self.unique_objects:
                        #if unique_objects exists, throw that triple!
                        if unique_objects: continue
                        self.unique_objects[t] = n_objects
                        n_objects += 1
                    if l not in self.unique_relationships:
                        #if unique_relationships exists, throw that triple!
                        if unique_relationships: continue
                        self.unique_relationships[l] = n_relationships
                        n_relationships += 1

                    #triplet is added!
                    self.triplets.append([
                        self.unique_objects[h],
                        self.unique_relationships[l],
                        self.unique_objects[t]
                    ])

            self.triplets = torch.LongTensor(self.triplets)
            self.n_objects = n_objects if not unique_objects else len(unique_objects)
            self.n_relationships = n_relationships if not unique_relationships else len(unique_relationships)

    def __len__(self):
        return self.triplets.shape[0]

    def __getitem__(self, idx):
        return self.triplets[idx]

def corrupted_head_or_tail(golden_batch, n_objects):
    #create corrupted triples!
    #only change either head or tail, simple random change (could match other golden triplet)
    #receive proper triplets
    hs, ls, ts = golden_batch[:,0], golden_batch[:, 1], golden_batch[:, 2]
    #coin flip
    coin = torch.randint(high=2, size=hs.size())
    #generate random objects
    random_ = torch.randint(high=n_objects, size=hs.size())
    #replace either head or tail!
    chs = torch.where(coin == 1, random_, hs)
    cts = torch.where(coin == 0, random_, ts)
    #return corrupted triplets batch...
    return torch.stack((chs, ls, cts), dim=1)

if __name__ == "__main__":
    '''
    This script receives a triplet file and outputs two json files
    entity2id and relationship2id in the directory of the given triplet file.

    To be used as: python triplets.py triplets_path
    '''
    #parser for all arguments!
    parser = argparse.ArgumentParser(description='Parsing triplets file...')

    #requirement arguments...
    parser.add_argument("triplets_path",
                        type=str, help="Path of file where triplets are saved...")

    args = parser.parse_args()

    #directory where triplets are stored...
    path=os.path.dirname(args.triplets_path)

    unique_objects = dict()
    unique_relationships = dict()

    n_objects = 0
    n_relationships = 0

    with open(args.triplets_path, 'r') as file:
        for line in file:
            #tab separated values!!!
            h, l, t = line.split('\t')
            #remove \n
            t = t[:-1]
            #we will encode the nodes and edges with unique integers!
            #this will match with the embedding Tensors that are used to contain
            #embeddings for our objects!
            if h not in unique_objects:
                unique_objects[h] = n_objects
                n_objects += 1
            if t not in unique_objects:
                unique_objects[t] = n_objects
                n_objects += 1
            if l not in unique_relationships:
                unique_relationships[l] = n_relationships
                n_relationships += 1

    #saving mappings...
    with open(path+'/entity2id.json', 'w') as f:
        json.dump(unique_objects, f)
    with open(path+'/relationship2id.json', 'w') as f:
        json.dump(unique_relationships, f)
