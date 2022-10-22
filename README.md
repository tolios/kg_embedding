# kg_embedding
This repo is designed for implementing various embedding algorithms for knowledge graphs.
It is under development (for my thesis). ((no GPU support (yet...)))

## id Mappings...
Before one does anything, it is required to produce mappings from raw triplets to ids.
To produce this, one should run the following code for their given training triplets...

```
python triplets.py triplet_path
```

This will produce 2 json files, one called entity2id and the other relationship2id, which contain
the aforementioned mappings! Those will be located in the same directory as the given triplet file.

## Training embeddings...

To train an embedding:
```
python main.py saved_embedding_folder \
       --train_data=train_path --val_data=valid_path \
       --lr=0.01 --train_batch_size=128 --epochs=50 --patience=20 \
       --algorithm=transe --emb_dim=100
```
python main.py --help (Will show only training arguments, without the specific class arguments)

To get all possible arguments for training, we have two types.
First, we have the general training arguments (i.e --train_data).
Secondly, having defined the --algorithm argument, one can fix
all the arguments that define the class Model inside the algorithms folder.
If one wants to train a TransE [[1]](#references) model, just add --algorithm=transe,
In general, one can define a personal type of embedding algorithm, by simply
creating a Python script (for example experiment.py) and then add --algorithm=experiment .

## Testing embeddings...

To test an embedding algorithm with metrics like hits@10 or Mean rank.
```
python test.py save_path algorithm mean_rank
```
save_path is the path of the saved model. The test results will be appended in a file called test.txt
located in the directory of the saved model.

Setting --filtering==True, one can get the results of the rankings with filtering.

To have the correct behaviour, one should define the arguments below:

* --test_data
* --train_data
* --valid_data

(Train and valid data are only used for the filtering of scores.)
```
python test.py --help (to get all arguments)
```

## A note...
This code does NOT guarantee the accurate implementation of the implemented algorithms, like TransE, TransH [[2]](#references) and RotatE [[3]](#references)...

## References
[1] [Bordes et al., "Translating embeddings for modeling multi- relational data," in Adv. Neural Inf. Process. Syst., 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)
[2] [Zhen Wang, Jianwen Zhang, et al. Knowledge Graph Embedding by Translating on Hyperplanes. Proceedings of AAAI, 2014.](https://ojs.aaai.org/index.php/AAAI/article/view/8870)
[3] [Zhiqing Sun et al., "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space"](https://openreview.net/pdf?id=HkgEQnRqYQ)
