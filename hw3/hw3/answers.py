r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.01
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.01
    reg = 0.01
    lr_momentum = 0.01
    lr_rmsprop = 0.0001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.03
    lr = 0.003
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
1. As one can see, the model without dropout achieves better train loss and accuracy results, which
comes as no surprise, since the dropout model randomly omits information about the input during
training, and is thus less capable of fitting the data. Since improved capability to fit the data
also means improved capability to overfit the data, the no-dropout model is more likely to overfit, 
which is demonstrated in the graphs - the dropout models perform better on the test set.

2. We observe that the higher dropout achieves better results during training, which we did not expect. 
It may be attributed to the fact that the higher dropout (0.8) introduces less randomness into the model
than the lower dropout (0.4, closer to 0.5) - neurons are very likely to be deactivated. Perhaps 
the reduced randomness allowed the model to compensate better for the lost neurons. In contrast, the lower dropout
 performs better on the test set, which could be expected, since it loses less information during training.
Overall we conclude that the lower dropout generalizes better, which may be due to the higher randomness it introduces,
randomness which helps avoid overfitting by "encouraging" neurons to be less reliant upon others and specialize less on
insignificant properties of the training data, and also due to the fact the it loses less information during training. 
            

"""

part2_q2 = r"""
When training a model with the cross-entropy loss function, it is possible for both the 
test accuracy and the test loss to increase for a few epochs. The loss depends not only
on whether the class with the maximum score is the correct class, but also on the model's level of confidence
that it is indeed the right class - i.e., the difference between the score of the true class to the other class scores.
Therefore, it may happen that for a few epochs, the number of correctly predicted classes remains the same 
or even slightly increases, but the confidence level decreases and thus the loss also increases.
For instance, the model can predict an input sample incorrectly in one epoch and correctly in the
following epoch, but the softmax score of the true class score remains the same or even decreases.

"""

part2_q3 = r"""
1. Gradient descent is an **optimization algorithm**. It finds local (and ideally global)
 optima by calculating the gradient of the loss function w.r.t the network parameters at the current point, and proceeding
  in the direction of the steepest descent (the opposite direction of the gradient). I.e., it is an algorithm that employs
the computation of gradients. On the other hand, backpropagation is an algorithm for **computing
these gradients**, by recursively going over the computation graph of the loss function backwards, from end to beginning,
and applying the chain rule at each step. 
2. In gradient descent, all input samples are used in order to calculate the gradient each time the parameters
are updated. In contrast, in stochastic gradient descent, a subset of the samples is picked randomly and the gradient
of the parameters is computed solely according to them. (If the subset is greater than one, it is called
mini-batch SGD). 
3. In SGD each step is much faster than a single GD step, since the computation of the gradient involves 
manipulating significantly less training instances, and it converges faster. This renders the SGD less computationally expensive
in terms of time and memory.
Furthermore, SGD can prove useful in cases where some directions are somewhat flat, and helps escape bad (shallow) 
local minima.
Lastly, SGD may generalize better since it usually does not find the optimum of the loss function and is thus
less capable of overfitting. 
4. A. this approach produces a gradient equivalent to the gradient produced by GD. 
Pytorch stores for each loss component (i.e., loss result
per batch) a computation graph. Once the backward function is invoked on the sum of the components,
the sum of the gradients of the components is calculated (since the derivative of a linear combination
of components is this linear combination of their gradients). The gradient of each loss component is computed according
 to its computation graphs, as it would be in GD. 
B. Pytorch maintains a computation graph for each loss tensor (for each batch), which is utilized
by the backward function. As part of this computation graph, the batches themselves
are stored. Hence, unfortunately the memory needed for this approach is linear with the size of the overall batches,
and after a few batches an out of memory
error might be incurred. 
"""
# ==============

# ==============
# Part 3 answers


def part3_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


def part3_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return start_seq, temperature


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
