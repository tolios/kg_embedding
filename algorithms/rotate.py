import math
import torch
from torch.nn.functional import normalize

class Model(torch.nn.Module):
    '''
    This custom class implements the RotatE algorithm. It contains all relevant
    embeddings and implements the loss function calculation. It also has some custom
    methods that are useful for predictions.
    '''
    def __init__(self, n_objects, n_relationships, emb_dim=100, margin=1.0, device=None):
        super(Model, self).__init__()
        self.n_objects = n_objects
        self.n_relationships = n_relationships
        self.emb_dim = emb_dim
        self.margin = margin
        self.device = device
        #creating embedding arrays...
        #cutom initializations
        #object_embedding! (real and imaginary parts!)
        self.object_embedding_re_ = torch.nn.Embedding(n_objects, emb_dim)
        self.object_embedding_re_.weight.data.uniform_(-6/math.sqrt(emb_dim),6/math.sqrt(emb_dim))
        self.object_embedding_im_ = torch.nn.Embedding(n_objects, emb_dim)
        self.object_embedding_im_.weight.data.uniform_(-6/math.sqrt(emb_dim),6/math.sqrt(emb_dim))
        #relationship_embedding!
        self.relationship_embedding_re_ = torch.nn.Embedding(n_relationships, emb_dim)
        self.relationship_embedding_re_.weight.data.uniform_(-6/math.sqrt(emb_dim),6/math.sqrt(emb_dim))
        self.relationship_embedding_im_ = torch.nn.Embedding(n_relationships, emb_dim)
        self.relationship_embedding_im_.weight.data.uniform_(-6/math.sqrt(emb_dim),6/math.sqrt(emb_dim))
        #used to implement loss! reduction = none, so it is used for outputing batch losses (later we sum them)
        self.criterion = torch.nn.MarginRankingLoss(margin=margin, reduction='none')

    def normalize(self):
        #with this method, all normalizations are performed.
        #To be used before mini-batch training in each epoch.
        #project to |ri| = 1
        rel_norm = torch.sqrt(self.relationship_embedding_re_.weight.data**2+self.relationship_embedding_im_.weight.data**2)
        self.relationship_embedding_re_.weight.data = self.relationship_embedding_re_.weight.data/rel_norm
        self.relationship_embedding_im_.weight.data = self.relationship_embedding_im_.weight.data/rel_norm
        #projecting object embeddings to have |e| = 1 (L2)
        obj_norm = torch.sum(self.object_embedding_re_.weight.data**2 + self.object_embedding_im_.weight.data**2, dim=-1)
        obj_norm = torch.sqrt(obj_norm).unsqueeze(1)
        self.object_embedding_re_.weight.data = self.object_embedding_re_.weight.data/obj_norm
        self.object_embedding_im_.weight.data = self.object_embedding_im_.weight.data/obj_norm

    def forward(self, correct: torch.LongTensor, corrupted: torch.LongTensor):
        #this method calculates the loss used in training!
        #receives batch of correct and corrupted triples and calculates loss for each batch member.
        #it also returns seperately the energies of correct and corrupted triplets, for recording!
        #to get the correct loss it need an aggregation, which can happen outside the forward call!
        h, l, t = correct[:, 0], correct[:, 1], correct[:, 2]
        #we throw out here the corrupted relationship, because its the same as the correct (transe alg)
        ch, ct = corrupted[:, 0], corrupted[:, 2]
        #now use embeddings!
        h_re = self.object_embedding_re_(h)
        h_im = self.object_embedding_im_(h)
        t_re = self.object_embedding_re_(t)
        t_im = self.object_embedding_im_(t)
        ch_re = self.object_embedding_re_(ch)
        ch_im = self.object_embedding_im_(ch)
        ct_re = self.object_embedding_re_(ct)
        ct_im = self.object_embedding_im_(ct)
        l_re = self.relationship_embedding_re_(l)
        l_im = self.relationship_embedding_im_(l)
        #calculate energies!!!
        E = self._energy(h_re, h_im, l_re, l_im, t_re, t_im)
        cE = self._energy(ch_re, ch_im, l_re, l_im, ct_re, ct_im)
        #now return loss!
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        loss = self.criterion(E, cE, target)
        return loss, E, cE

    def predict(self, h: torch.LongTensor, l: torch.LongTensor, t: torch.LongTensor):
        #this method is used to calculate the energy for a batch of triplets
        #receives them in LongTensor form (ids)...
        h_re = self.object_embedding_re_(h)
        h_im = self.object_embedding_im_(h)
        l_re = self.relationship_embedding_re_(l)
        l_im = self.relationship_embedding_im_(l)
        t_re = self.object_embedding_re_(t)
        t_im = self.object_embedding_im_(t)
        #return energy!
        return self._energy(h_re, h_im, l_re, l_im, t_re, t_im)

    def _energy(self, h_re: torch.Tensor, h_im: torch.Tensor,
                      l_re: torch.Tensor, l_im: torch.Tensor,
                      t_re: torch.Tensor, t_im: torch.Tensor):
        #calculates distance (energy) given h, l, t embeddings of triplets (real and imaginary)!
        #calculate h*r real and imaginary components
        _re = (h_re*l_re)-(h_im*l_im)
        _im = (h_re*l_im)+(h_im*l_re)
        #minus tail!
        _re = _re - t_re
        _im = _im - t_im
        #sum and sqrt (only L2), then return
        return torch.sqrt(torch.sum((_re**2) + (_im**2), dim=-1))

    def save(self, path):
        #save model!
        torch.save({
            'n_objects': self.n_objects,
            'n_relationships': self.n_relationships,
            'emb_dim': self.emb_dim,
            'margin': self.margin,
            'state_dict': self.state_dict()}, path)

    def load(self, path, device = None):
        #load model (into gpu if you want)
        checkpoint = torch.load(path)
        n_objects = checkpoint['n_objects']
        n_relationships = checkpoint['n_relationships']
        emb_dim = checkpoint['emb_dim']
        margin = checkpoint['margin']
        model = Model(n_objects, n_relationships, emb_dim=emb_dim, margin=margin, device=device)
        model.load_state_dict(checkpoint['state_dict'])
        self = model
        return self
