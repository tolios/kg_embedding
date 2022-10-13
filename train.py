import time
import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt

def training(model: torch.nn.Module, train: Dataset, val: Dataset,
    epochs = 50, batch_size = 1024, val_batch_size = 1024,
    lr = 0.01, weight_decay = 0.0005, patience = -1):
    '''
    Iplementation of training. Receives embedding model, dataset of training and val data!.
    Returns trained model, training losses, Uncorrupted and Corrupted energies.
    '''
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=val_batch_size, shuffle=True)
    #optimizers
    optimizer = SGD(model.parameters(), lr = lr, weight_decay = weight_decay)
    #training begins...
    t_start = time.time()
    losses = []
    energies = []
    c_energies = []
    val_energies = []
    #for early stopping!
    #start with huge number!
    lowest_val_energy = 1e5
    stop_counter = 1
    epoch_stop = 0 #keeps track of last epoch of checkpoint...
    #training ...
    print('Training begins ...')
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_E = 0.0
        running_cE = 0.0
        #perform normalizations before entering the mini-batch.
        model.normalize()
        for i, batch in enumerate(train_loader):
            #receive proper triplets
            hs, ls, ts = batch[:,0], batch[:, 1], batch[:, 2]
            #create corrupted triples!
            coin = torch.randint(high=2, size=hs.size())
            random_ = torch.randint(high=train.n_objects, size=hs.size())
            chs = torch.where(coin == 1, random_, hs)
            cts = torch.where(coin == 0, random_, ts)
            #make triplets...
            corrupted = torch.stack((chs, ls, cts), dim=1)
            #calculate loss...
            loss, E, cE = model(batch, corrupted)
            #zero out gradients...
            optimizer.zero_grad()
            #loss backward
            loss.sum().backward()
            #update parameters!
            optimizer.step()
            #getting losses...
            running_loss += loss.mean().data.item()
            running_E += E.mean().data.item()
            running_cE += cE.mean().data.item()
        #calculating val energy....
        with torch.no_grad():
            running_val_E = 0.0
            for j, batch in enumerate(val_loader):
                #receive proper triplets
                hs, ls, ts = batch[:,0], batch[:, 1], batch[:, 2]
                #calculate validation energies!!!
                running_val_E += model.predict(hs, ls, ts).mean().data.item()
        #print results...
        print('Epoch: ', epoch, ', loss: ', "{:.4f}".format(running_loss/i),
            ', energy: ', "{:.4f}".format(running_E/i),
            ', corrupted energy: ', "{:.4f}".format(running_cE/i),
            ', val_energy: ', "{:.4f}".format(running_val_E/j),
            ', time: ', "{:.4f}".format((time.time()-t_start)/60), 'min(s)')
        #collecting loss and energies!
        losses.append(running_loss/i)
        energies.append(running_E/i)
        c_energies.append(running_cE/i)
        val_energies.append(running_val_E/j)
        #implementation of early stop using val_energy (fastest route (could use mean_rank for example))
        if patience != -1:
            if lowest_val_energy >= running_val_E/j:
                #setting new energy!
                lowest_val_energy = running_val_E/j
                #save model checkpoint...
                model.save('./checkpoint.pth.tar')
                epoch_stop = epoch
                #"zero out" counter
                stop_counter = 1
            else:
                if stop_counter >= patience:
                    #no more patience...early stopping!
                    print('Early stopping at epoch:', epoch)
                    print('Loading from epoch:', epoch_stop)
                    #load model from previous checkpoint!
                    model.load('./checkpoint.pth.tar')
                    break
                else:
                    #be patient...
                    stop_counter += 1
                    #if in the end of training load from checkpoint!
                    if epoch == epochs:
                        print('Finished during early stopping...')
                        print('Loading from epoch:', epoch_stop)
                        #load model from previous checkpoint!
                        model.load('./checkpoint.pth.tar')
        else:
            epoch_stop = epochs
    ## If checkpoint exists, delete it ##
    if os.path.isfile('./checkpoint.pth.tar'):
        os.remove('./checkpoint.pth.tar')

    print('Training ends ...')
    #returning model as well as all energies until early stop, if it happened!
    return model, losses[:epoch_stop], energies[:epoch_stop], c_energies[:epoch_stop], val_energies[:epoch_stop]
