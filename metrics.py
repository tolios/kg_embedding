import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def tail_energies(triplet_batch: torch.LongTensor, model:torch.nn.Module):
    #true triplets
    h, l, t = triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]
    #
    h = model.object_embedding_(h)
    l = model.relationship_embedding_(l)
    h, l = torch.unsqueeze(h, dim=1), torch.unsqueeze(l, dim=1)
    h, l = h.expand(h.shape[0], model.n_objects, model.emb_dim), l.expand(l.shape[0], model.n_objects, model.emb_dim)
    # emb_t: [batch_size, N, embed_size]
    emb_t = model.object_embedding_.weight.data.expand(t.shape[0], model.n_objects, model.emb_dim)
    #calculate energies!
    E = model._energy(h, l, emb_t)
    #return energies and corresponding tails!
    return E, t

def head_energies(triplet_batch: torch.LongTensor, model:torch.nn.Module):
    #true triplets
    h, l, t = triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]
    #
    t = model.object_embedding_(t)
    l = model.relationship_embedding_(l)
    t, l = torch.unsqueeze(t, dim=1), torch.unsqueeze(l, dim=1)
    t, l = t.expand(t.shape[0], model.n_objects, model.emb_dim), l.expand(l.shape[0], model.n_objects, model.emb_dim)
    # emb_t: [batch_size, N, embed_size]
    emb_h = model.object_embedding_.weight.data.expand(h.shape[0], model.n_objects, model.emb_dim)
    #calculate energies!
    E = model._energy(emb_h, l, t)
    #return energies and corresponding heads!
    return E, h

def hits_at_N(data: torch.utils.data.Dataset, model: torch.nn.Module, N = 10, batch_size = 64):
    '''
    Evaluates hits@N metric for given dataset and model.
    data: Triplets object
    model: Model object of given embedding strategy!
    '''
    with torch.no_grad():
        hits = 0
        N_triples = len(data)
        loader = DataLoader(data, batch_size = batch_size)
        for batch in tqdm(loader, desc=f'hits@{N} calculation'):
            #calculating head and tail energies for prediction
            head_E, h = head_energies(batch, model)
            tail_E, t = tail_energies(batch, model)
            h, t = h.view(-1, 1), t.view(-1, 1)
            #calculating indices for topk...
            _, h_indices = torch.topk(head_E, N, dim=1, largest=False)
            _, t_indices = torch.topk(tail_E, N, dim=1, largest=False)
            #summing hits...
            hits += torch.sum(torch.eq(h_indices, h)).item()+torch.sum(torch.eq(t_indices, t)).item()
        #return total hits over 2*N_triples!
        return hits/(2*N_triples)

def mean_rank(data: torch.utils.data.Dataset, model: torch.nn.Module, batch_size = 64):
    '''
    Evaluates mean_rank metric for given data, embeddings and energy.
    data: Triplets object
    object_embedding, relationship_embedding: Embedding object
    energy: function that determines energy for triplet batch embeddings!
    '''
    with torch.no_grad():
        mean = 0.
        N_triples = len(data)
        loader = DataLoader(data, batch_size = batch_size)
        for batch in tqdm(loader, desc=f'mean rank calculation'):
            #calculating head and tail energies for prediction
            head_E, h = head_energies(batch, model)
            tail_E, t = tail_energies(batch, model)
            h, t = h.view(-1, 1), t.view(-1, 1)
            #calculating indices for sorting...
            _, h_indices = torch.sort(head_E, dim = 1)
            _, t_indices = torch.sort(tail_E, dim = 1)
            #adding index places...
            mean += torch.sum(torch.eq(h_indices,h).nonzero()[:, 1]).item()
            mean += torch.sum(torch.eq(t_indices,t).nonzero()[:, 1]).item()

        return mean/(2*N_triples)
