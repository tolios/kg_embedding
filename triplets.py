import torch
from torch.utils.data import Dataset
import random

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
            self.n_objects = n_objects
            self.n_relationships = n_relationships 

    def __len__(self):
        return self.triplets.shape[0]

    def __getitem__(self, idx):
        return self.triplets[idx]

def corruptor(triplets):
    #receives a triplets dataset and produces two dictionaries!
    #one containing all corrupted heads of each relation
    #the other all the tails!
    head_ = dict()
    tail_ = dict()
    total = set()
    for h, l, t in triplets.triplets:
        h, l, t = int(h), int(l), int(t)
        #add heads for each relation
        if l not in head_:
            head_[l] = {h}
        else:
            head_[l].add(h)
        #add tails for each relation
        if l not in tail_:
            tail_[l] = {t}
        else:
            tail_[l].add(t)
        #find all objects!
        total.add(h)
        total.add(t)
    #generate corrupted triple dictionary...
    for h in head_:
        head_[h] = total - head_[h]
    for t in tail_:
        tail_[t] = total - tail_[t]

    #returns dictionary containing corrupted head and tail
    return head_, tail_

def corrupted_triplet(triplet, corrupted_heads, corrupted_tails):
    #receive a corrupted triplet and output a random corrupted one!
    #returns corrupted triple!
    h, l, t = triplet
    h, l, t = int(h), int(l), int(t)

    coin = random.randint(0, 1)
    if coin:
        #replace head with corrupted one
        corr_h = random.choice(tuple(corrupted_heads[l]))
        return [corr_h, l, t]
    else:
        #replace tail with corrupted one
        corr_t = random.choice(tuple(corrupted_tails[l]))
        return [h, l, corr_t]
