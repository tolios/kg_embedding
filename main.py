import time
import os
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from train import *
from triplets import *
from metrics import *
import matplotlib.pyplot as plt
import argparse
import importlib
import inspect

#parser for all arguments!
parser = argparse.ArgumentParser(description='Training knowledge graph embeddings...',
                                 epilog='''
                                    NOTE: You can also add as arguments the kwargs in the Model class,
                                    defined inside the algorithms folder. For example, if --algorithm=transe,
                                    then all kwargs defined in the transe.Model class, can be changed i.e --norm=1
                                        ''')
#requirement arguments...
parser.add_argument("save_path",
                    type=str, help="Directory where model is saved")
#optional arguments...
parser.add_argument("--algorithm",
                    default='transe',
                    type=str, help="Embedding algorithm (stored in algorithms folder!)")
parser.add_argument("--seed",
                    default=42,
                    type=int, help="Seed for randomness")
parser.add_argument("--train_data",
                    default='./FB15k/freebase_mtr100_mte100-train.txt',
                    type=str, help="Path to training data")
parser.add_argument("--val_data",
                    default='./FB15k/freebase_mtr100_mte100-valid.txt',
                    type=str, help="Path to validation data")
parser.add_argument("--epochs",
                    default=25,
                    type=int, help="Number of training epochs")
parser.add_argument("--train_batch_size",
                    default=1024,
                    type=int, help="Training data batch size")
parser.add_argument("--val_batch_size",
                    default=1024,
                    type=int, help="Validation data batch size")
parser.add_argument("--lr",
                    default=0.001,
                    type=float, help="Learning rate")
parser.add_argument("--weight_decay",
                    default=0.0,
                    type=float, help="Weight decay")
parser.add_argument("--patience",
                    default=-1,
                    type=int, help="Patience for early stopping")
parser.add_argument("--val_calc",
                    default = False,
                    type=bool, help="Calculation of validation metrics after training...")
#parse known and unknown args!!!
args, unknown = parser.parse_known_args()

#custom parsed arguments from Model kwargs!!!
#given module... algorithm argument
module = importlib.import_module('algorithms.'+args.algorithm, ".")
#module.Model keyword args!
spec_args = inspect.getfullargspec(module.Model)
values = spec_args.defaults
custom_args = spec_args.args[-len(values):]
#make arg dictionary
model_args = {x:y for x, y in zip(custom_args, values)}
for arg in model_args:
    #adding Model keyword arguments to the parser!!!
    parser.add_argument("--"+arg,default=model_args[arg],
                            type = type(model_args[arg]))

#finds all arguments...
args = parser.parse_args()

#seeds
torch.manual_seed(args.seed)

#configs
TRAIN_PATH = args.train_data
VAL_PATH = args.val_data
EPOCHS = args.epochs
BATCH_SIZE = args.train_batch_size
VAL_BATCH_SIZE = args.val_batch_size
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay
PATIENCE = args.patience
SAVE_PATH = args.save_path
algorithm = args.algorithm
val_calc = args.val_calc

#directory where triplets are stored... as well as ids!
id_dir=os.path.dirname(TRAIN_PATH)

#loading ids...
with open(id_dir+'/entity2id.json', 'r') as f:
    unique_objects = json.load(f)
with open(id_dir+'/relationship2id.json', 'r') as f:
    unique_relationships = json.load(f)

#now update model dictionary with possible given values!
for arg in model_args:
    model_args[arg] = vars(args)[arg]

#data
#training
train = Triplets(path = TRAIN_PATH, unique_objects = unique_objects,
                        unique_relationships = unique_relationships)
#validation
val =  Triplets(path = VAL_PATH, unique_objects = unique_objects,
                        unique_relationships = unique_relationships)
#define trainable embeddings!
model = module.Model(len(unique_objects), len(unique_relationships), **model_args)

start = time.time()
#training begins...
model, losses, energies, c_energies, val_energies = training(model, train, val,
                epochs = EPOCHS, batch_size = BATCH_SIZE, val_batch_size = VAL_BATCH_SIZE,
                lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY, patience = PATIENCE)
end = time.time()
if val_calc:
    print('Calculating validation scores!')
    #calculating validation hits@10 and mean rank!
    hits_at = hits_at_N(val, model, batch_size = 256, N=10)*100
    m_rank = mean_rank(val, model, batch_size = 128)

    print('Validation scores: ')
    print(f'hits@10 = {hits_at} %')
    print(f'mean rank = {m_rank}')

#actuall epochs
actual_epochs = len(energies)

#plot!
plt.plot([x for x in range(actual_epochs)], losses, label='Training loss')
plt.plot([x for x in range(actual_epochs)], energies, label='Uncorrupted energy')
plt.plot([x for x in range(actual_epochs)], c_energies, label='Corrupted energy')
plt.plot([x for x in range(actual_epochs)], val_energies, label='Validation energy')
plt.legend()
plt.xlabel('Epoch(s)')
plt.title('Loss per Epoch')

#save model!
#create folder containing embeddings
os.makedirs(SAVE_PATH)
model.save(SAVE_PATH+'/model.pth.tar')
#savefig
plt.savefig(SAVE_PATH+'/training_fig.png')
#save train configuration!
with open(SAVE_PATH+'/train_config.txt', 'w') as file:
    file.write(f'ALGORITHM: {algorithm}\n')
    file.write(f'SEED: {args.seed}\n')
    file.write(f'TRAIN_PATH: {TRAIN_PATH}\n')
    file.write(f'VAL_PATH: {VAL_PATH}\n')
    file.write(f'NUM_EMBEDDINGS_OBJECT: {train.n_objects}\n')
    file.write(f'NUM_EMBEDDINGS_RELATIONSHIP: {train.n_relationships}\n')
    file.write(f'EPOCHS: {EPOCHS}\n')
    file.write(f'ACTUAL_EPOCHS: {actual_epochs}\n')
    file.write(f'PATIENCE: {PATIENCE}\n')
    file.write(f'BATCH_SIZE: {BATCH_SIZE}\n')
    file.write(f'VAL_BATCH_SIZE: {VAL_BATCH_SIZE}\n')
    file.write(f'LEARNING_RATE: {LEARNING_RATE}\n')
    file.write(f'WEIGHT_DECAY: {WEIGHT_DECAY}\n')
    if val_calc:
        file.write(f'hits@10: {hits_at}%\n')
        file.write(f'mean_rank: {m_rank}\n')
    file.write('Model args:\n')
    for arg in model_args:
        file.write(f'{arg}: {model_args[arg]}\n')
    file.write(f'Training time: {"{:.4f}".format((end-start)/60)} min(s)')
