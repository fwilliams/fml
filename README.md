FML (Francis' Machine-Learnin' Library)
===================================

This repository is a collection of PyTorch tools for machine learning. Currently it includes 
* A numerically stable implementation of the [Sinkhorn Algorithm](https://arxiv.org/abs/1306.0895) for point sets in any dimension
* A vectorized implementation of the Chamfer Distance between point sets in any dimension

## Installation Instructions

### With conda (recommended)
On Linux, Simply run:
```
conda install -c conda-forge fml
```

On Windows and Mac, you may need to add the PyTorch Channel before installing:
```
conda config --add channels pytorch
conda install -c conda-forge fml
```

### With pip (not recommended)
Simply run:
```
pip install git+https://github.com/fwilliams/fml
```



## Library Structure
The structure of the library is similar to PyTorch. There is a `fml.functional` module which includes a functional interface for utilities and an `fml.nn` which includes PyTorch module implementations of utilities.

## Examples

### Computing the loss between two evenly weighted point sets
```python
import torch
from fml.nn import SinkhornLoss

minibatch_size = 3
set_size = 10
point_dim = 4

# Create two minibatches of point sets where each batch item set_a[k, :, :] is a set of `set_size` points
set_a = torch.rand([minibatch_size, set_size, point_dim])
set_b = torch.rand([minibatch_size, set_size, point_dim])

# Create a loss function module with default parameters. See the class documentation for optional parameters.
loss_fun = SinkhornLoss()

# Compute the loss between each pair of sets in the minibatch
# loss is a tensor with [minibatch_size] elements which can be backpropagated through
loss = loss_fun(set_a, set_b)
```

### Computing the loss between two non evenly weighted point sets
```python
import torch
from fml.nn import SinkhornLoss

minibatch_size = 3
set_size = 10
point_dim = 4

# Create two minibatches of point sets where each batch item set_a[k, :, :] is a set of `set_size` points
set_a = torch.rand([minibatch_size, set_size, point_dim])
set_b = torch.rand([minibatch_size, set_size, point_dim])

# Generate weights which sum to 1 for each set
# Note that zero weights are the same as not including a set element
weights_a = torch.rand([minibatch_size, set_size])
weights_a /= torch.sum(weights_a, axis=1) 

weights_b = torch.rand([minibatch_size, set_size])
weights_b /= torch.sum(weights_b, axis=1) 

# Create a loss function module with default parameters. See the class documentation for optional parameters.
loss_fun = SinkhornLoss()

# Compute the loss between each pair of sets in the minibatch
# loss is a tensor with [minibatch_size] elements which can be backpropagated through
loss = loss_fun(set_a, set_b)
```


### Computing the Chamfer Distance between point sets
```python
import torch
from fml.nn import ChamferLoss

minibatch_size = 3
set_size = 10
point_dim = 4

# Create two minibatches of point sets where each batch item set_a[k, :, :] is a set of `set_size` points
set_a = torch.rand([minibatch_size, set_size, point_dim])
set_b = torch.rand([minibatch_size, set_size, point_dim])

# Create a loss function module.
loss_fun = ChamferLoss()

# Compute the loss between each pair of sets in the minibatch
# loss is a tensor with [minibatch_size] elements which can be backpropagated through
loss = loss_fun(set_a, set_b)
```

### Computing pairwise distances between point sets
```python
import torch
from fml.functional import pairwise_distances

minibatch_size = 3
set_size = 10
point_dim = 4

# Create two minibatches of point sets where each batch item set_a[k, :, :] is a set of `set_size` points
set_a = torch.rand([minibatch_size, set_size, point_dim])
set_b = torch.rand([minibatch_size, set_size, point_dim])

# Compute the pairwise distances between each pair of sets in the minibatch
# distances is a tensor of shape [minibatch_size, set_size, set_size] where each
# disances[k, i, j] = ||set_a[k, i] - set_b[k, j]||^2
distances = pairwise_distances(set_a, set_b)
```
