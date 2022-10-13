#Embedding algorithm

To construct an embedding algorithm, one must construct a class which inherits form torch.nn.Module.
It needs the following methods:

* normalize
* forward
* predict
* save
* load

##normalize

This method is used to do (hard) normalizations (for example set all entity embeddings to have L2 norm equal to 1.).
Usually embedding algorithms do these normalizations before accessing each batch in the start of all epochs.
This implementation does as well.

##forward

This method receives a batch of correct and corrupted LongTensor triples and outputs a tuple (Loss, energy, corrupted energy).
Here one must implement the loss part of the algorithm. All embedding algorithms require both golden and corrupted triples.
Furthermore, the goal of most embedding algorithms is to minimize some sort of energy function for the golden triples,
while maximizing for the corrupted ones. Therefore, the forward method outputs also the energy and the corrupted energy for
all batch members. Here there are implemented also (soft) normalizations or rather regularizations.
(Note: we calculate loss, energy and corrupted energy for each batch, so they should be of shape (batch_size,))

##predict

This method receives head, relation and tail LongTensor batches and outputs the aforementioned energies.
