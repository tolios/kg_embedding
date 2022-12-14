import math
import torch
from torch.nn.functional import normalize

class Model(torch.nn.Module):
    '''
    This custom class implements the TransH algorithm. It contains all relevant
    embeddings and implements the loss function calculation. It also has some custom
    methods that are useful for predictions.
    '''
    def __init__(self, n_objects, n_relationships, norm=2, emb_dim=100, margin=1.0, c=0.25, eps = 0.0001, device=None):
        super(Model, self).__init__()
        self.n_objects = n_objects
        self.n_relationships = n_relationships
        self.norm = norm
        self.emb_dim = emb_dim
        self.margin = margin
        self.device = device
        #creating embedding arrays...
        #cutom initializations
        #object_embedding!
        self.object_embedding_ = torch.nn.Embedding(n_objects, emb_dim)
        self.object_embedding_.weight.data.uniform_(-6/math.sqrt(emb_dim),6/math.sqrt(emb_dim))
        #relationship_embedding!
        self.relationship_embedding_ = torch.nn.Embedding(n_relationships, emb_dim)
        self.relationship_embedding_.weight.data.uniform_(-6/math.sqrt(emb_dim),6/math.sqrt(emb_dim))
        #normalize relationship_embedding...
        self.relationship_embedding_.weight.data = normalize(self.relationship_embedding_.weight.data, dim = 1, p = norm)
        #relationship projection!
        self.relationship_projection_ = torch.nn.Embedding(n_relationships, emb_dim)
        self.relationship_projection_.weight.data.uniform_(-6/math.sqrt(emb_dim),6/math.sqrt(emb_dim))
        #used to implement loss! reduction = none, so it is used for outputing batch losses (later we sum them)
        self.criterion = torch.nn.MarginRankingLoss(margin=margin, reduction='none')
        #finally adding a parameter that weights the regularizer terms!
        self.c = c
        self.eps = eps #used in enforcing relationship_projection is normal to relationship_embedding

    def normalize(self):
        #with this method, all normalizations are performed.
        #To be used before mini-batch training in each epoch.
        self.relationship_projection_.weight.data = normalize(self.relationship_projection_.weight.data,dim = 1, p = self.norm)
        self.object_embedding_.weight.data = normalize(self.object_embedding_.weight.data, dim = 1, p = self.norm)

    def forward(self, correct: torch.LongTensor, corrupted: torch.LongTensor):
        #this method calculates the loss used in training!
        #receives batch of correct and corrupted triples and calculates loss for each batch member.
        #it also returns seperately the energies of correct and corrupted triplets, for recording!
        #to get the correct loss it need an aggregation, which can happen outside the forward call!
        #now we find all embeddings!
        h, l, t = correct[:, 0], correct[:, 1], correct[:, 2]
        #we throw out here the corrupted relationship, because its the same as the correct (transe alg)
        ch, ct = corrupted[:, 0], corrupted[:, 2]
        #now use embeddings!
        h = self.object_embedding_(h)
        t = self.object_embedding_(t)
        ch = self.object_embedding_(ch)
        ct = self.object_embedding_(ct)
        w = self.relationship_projection_(l)
        l = self.relationship_embedding_(l)
        #calculate energies!!!
        E = self._energy(h, l, t, w)
        cE = self._energy(ch, l, ct, w)
        #now return loss!
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        loss = self.criterion(E, cE, target)
        #adding regularizer terms!!!
        reg_h = self.c*torch.relu(h.norm(self.norm,dim=1)**2 - 1)
        reg_t = self.c*torch.relu(t.norm(self.norm,dim=1)**2 - 1)
        reg_wl = self.c*torch.relu(((torch.sum(w*l, axis = 1)**2)/(l.norm(self.norm,dim=1)**2))-self.eps)
        
        loss += reg_h + reg_t + reg_wl
        return loss, E, cE

    def predict(self, h: torch.LongTensor, l: torch.LongTensor, t: torch.LongTensor):
        #this method is used to calculate the energy for a batch of triplets
        #receives them in LongTensor form (ids)...
        w = self.relationship_projection_(l)
        h = self.object_embedding_(h)
        l = self.relationship_embedding_(l)
        t = self.object_embedding_(t)
        #return energy!
        return self._energy(h,l,t,w)

    def _energy(self, h: torch.Tensor, l: torch.Tensor, t: torch.Tensor, w: torch.Tensor):
        #calculates distance (energy) given h, l, t embeddings of triplets! (with projection w)
        return (h+l-t-(w*torch.sum((h-t)*w,dim=-1).unsqueeze(-1))).norm(p=self.norm,dim=-1)

    def save(self, path):
        #save model!
        torch.save({
            'n_objects': self.n_objects,
            'n_relationships': self.n_relationships,
            'norm': self.norm,
            'emb_dim': self.emb_dim,
            'margin': self.margin,
            'state_dict': self.state_dict()}, path)

    def load(self, path, device = None):
        #load model (into gpu if you want)
        checkpoint = torch.load(path)
        n_objects = checkpoint['n_objects']
        n_relationships = checkpoint['n_relationships']
        norm = checkpoint['norm']
        emb_dim = checkpoint['emb_dim']
        margin = checkpoint['margin']
        model = Model(n_objects, n_relationships, norm=norm, emb_dim=emb_dim, margin=margin, device=device)
        model.load_state_dict(checkpoint['state_dict'])
        self = model
        return self
