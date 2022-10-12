import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Filter:
    '''
    This class receives train, val and test data so as to create a filter function!
    '''
    def __init__(self, train, val, test, big = 10e5):
        self.n_objects = train.n_objects
        self.dict_ = self._create_dict(train, val, test)
        self.big = big

    def mask(self, batch):
        h, r, t = batch[:,0], batch[:,1], batch[:,2]
        batch_mask = []
        for h_, r_, t_ in zip(h, r, t):
            batch_mask.append(self._mask(h_.item(),r_.item(), t_.item()))
        return torch.stack(batch_mask, dim=0)


    def _mask(self, h:int, r:int, t:int):
        #for a single triple
        m = torch.ones(self.n_objects)
        #find all the rest golden tails
        golden = self.dict_[(h, r)]
        #without t
        golden = golden - {t}
        #make mask
        for t_ in golden:
            m[t_] = self.big
        return m

    def _create_dict(self, train, val, test):
        #this function creates a dictionary which hashes (head, relation) pair
        #and contains the set of corresponding tails that exist!
        dict_ = dict()
        #train
        for triple in train:
            h, r, t = triple
            h, r, t = h.item(), r.item(), t.item()
            if (h, r) not in dict_:
                dict_[(h,r)] = {t}
            else:
                dict_[(h,r)].add(t)
        #test
        for triple in val:
            h, r, t = triple
            h, r, t = h.item(), r.item(), t.item()
            if (h, r) not in dict_:
                dict_[(h,r)] = {t}
            else:
                dict_[(h,r)].add(t)
        #val
        for triple in test:
            h, r, t = triple
            h, r, t = h.item(), r.item(), t.item()
            if (h, r) not in dict_:
                dict_[(h,r)] = {t}
            else:
                dict_[(h,r)].add(t)
        return dict_

def tail_energies(triplet_batch: torch.LongTensor, model:torch.nn.Module):
    #true triplets
    h, l, t = triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]
    #strategy is to compare the same h,l with every tail
    h, l = torch.unsqueeze(h, dim=1), torch.unsqueeze(l, dim=1)
    h, l = h.expand(h.shape[0], model.n_objects), l.expand(l.shape[0], model.n_objects)
    #every tail per batch (expand)
    emb_t = torch.arange(0, model.n_objects).expand(t.shape[0], model.n_objects)
    #calculate energies!
    E = model.predict(h, l, emb_t)
    #return energies and corresponding tails!
    return E, t

def hits_at_N(data: torch.utils.data.Dataset, model: torch.nn.Module, N = 10, batch_size = 64, filter = None):
    '''
    Evaluates hits@N metric for given dataset and model.
    data: Triplets object
    model: Model object of given embedding strategy!
    filter: Filter object, that creates appropriate masks
    '''
    mode = 'Filt.' if filter else 'Raw'
    with torch.no_grad():
        hits = 0
        N_triples = len(data)
        loader = DataLoader(data, batch_size = batch_size)
        for batch in tqdm(loader, desc=f'{mode} hits@{N} calculation'):
            #calculating head and tail energies for prediction
            tail_E, t = tail_energies(batch, model)
            if filter:
                #use Filter object mask method
                tail_E = tail_E*filter.mask(batch)
            t = t.view(-1, 1)
            #calculating indices for topk...
            _, t_indices = torch.topk(tail_E, N, dim=1, largest=False)
            #summing hits...
            zero_tensor = torch.tensor([0])
            one_tensor = torch.tensor([1])
            hits += torch.where(torch.eq(t_indices, t), one_tensor, zero_tensor).sum().item()
        #return total hits over 2*N_triples!
        return hits/(N_triples)

def mean_rank(data: torch.utils.data.Dataset, model: torch.nn.Module, batch_size = 64, filter = None):
    '''
    Evaluates mean_rank metric for given data, embeddings and energy.
    data: Triplets object
    object_embedding, relationship_embedding: Embedding object
    energy: function that determines energy for triplet batch embeddings!
    '''
    mode = 'Filt.' if filter else 'Raw'
    with torch.no_grad():
        mean = 0.
        N_triples = len(data)
        loader = DataLoader(data, batch_size = batch_size)
        for batch in tqdm(loader, desc=f'{mode} mean rank calculation'):
            #calculating head and tail energies for prediction
            tail_E, t = tail_energies(batch, model)
            if filter:
                #use Filter object mask method
                tail_E = tail_E*filter.mask(batch)
            t = t.view(-1, 1)
            #calculating indices for sorting...
            _, t_indices = torch.sort(tail_E, dim = 1)
            #adding index places...
            mean += torch.sum(torch.eq(t_indices,t).nonzero()[:, 1]).item()
        return mean/(N_triples)
