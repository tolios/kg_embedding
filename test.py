import os
import argparse
import importlib
from triplets import *
from metrics import *

#parser for all arguments!
parser = argparse.ArgumentParser(description='Testing knowledge graph embeddings...')

#requirement arguments...
parser.add_argument("save_path",
                    type=str, help="Directory where model is saved")

parser.add_argument("algorithm",
                    type=str, help="Embedding algorithm of saved model to be tested!")

parser.add_argument("metric",
                    choices=['mean_rank', 'hits@'],
                    type=str, help="Metric to be used for testing!")

#optional requirements!
parser.add_argument("--N",
                    default=10,
                    type=int, help="hits@N N. Only used when hits@ is used as argument in metric")

parser.add_argument("--test_data",
                    default='./FB15k_237/test.txt',
                    type=str, help="Path to test data")

parser.add_argument("--filtering",
                    default = False,
                    type=bool, help="Filter out true triplets, that artificially lower the scores...")

parser.add_argument("--train_data",
                    default='./FB15k_237/train.txt',
                    type=str, help="Path to training data (used for filtering)")

parser.add_argument("--val_data",
                    default='./FB15k_237/valid.txt',
                    type=str, help="Path to validation data (used for filtering)")

parser.add_argument("--big",
                    default='10e5',
                    type=float, help="Value of mask, so as to filter out golden triples")

parser.add_argument("--batch_size",
                    default=64,
                    type=int, help="Test batch size")

parser.add_argument("--seed",
                    default=42,
                    type=int, help="Seed for randomness")

#finds all arguments...
args = parser.parse_args()

SAVE_PATH = args.save_path
ALGORITHM = args.algorithm
metric_ = args.metric
filtering = args.filtering
test_data = args.test_data
train_data = args.train_data
val_data = args.val_data
N = args.N
batch_size = args.batch_size

#seeds
torch.manual_seed(args.seed)

#import algorithm
module = importlib.import_module('algorithms.'+args.algorithm, ".")

#data
#training
train = Triplets(path = train_data)

unique_objects = train.unique_objects
unique_relationships = train.unique_relationships
#validation
val =  Triplets(path = val_data, unique_objects = unique_objects,
                        unique_relationships = unique_relationships)
#test
test =  Triplets(path = test_data, unique_objects = unique_objects,
                        unique_relationships = unique_relationships)

#load model...
model = module.Model(train.n_objects, train.n_relationships)
model = model.load(SAVE_PATH)

if filtering:
    filter = Filter(train, val, test, big = args.big)
else:
    filter = None

path=os.path.dirname(SAVE_PATH)

if metric_ == 'mean_rank':
    result = mean_rank(test, model, filter = filter, batch_size = batch_size)
    with open(path+"/test.txt", "a") as myfile:
        s = 'Filt.' if filtering else 'Raw'
        myfile.write(f'{s} mean rank = {"{:.2f}".format(result)}\n')
elif metric_ == 'hits@':
    result = hits_at_N(test, model, N=N, filter = filter, batch_size = batch_size)
    with open(path+"/test.txt", "a") as myfile:
        s = 'Filt.' if filtering else 'Raw'
        myfile.write(f'{s} hits@{N} = {"{:.2f}".format(result*100)}%\n')
else:
    raise
