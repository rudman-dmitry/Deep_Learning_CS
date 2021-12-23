r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 3
    activation = "tanh"
    #out_activation = "logsoftmax"
    out_activation = "tanh"
    hidden_dims = 7
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part1_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    #loss_fn = torch.nn.NLLLoss()
    loss_fn = torch.nn.CrossEntropyLoss()
    #lr = 0.1
    lr= 0.03
    momentum = 0.4
    weight_decay = 0.01
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part1_q1 = r"""
**Your answer:**
1. Optimization error - considering the plotted curves, we note that the model achieves 92.5% accuracy on the train set
and therefore we conclude that it has a somewhat high optimization error and we would take measures to decrease it.
2. Generalization error - the test curve is relatively close to the test curve. However, considering the decision plot boundary, 
we see that rotating the upper arm of the z-shape slightly to the left can yield better results on the test set, and therefore
we believ that the model has a somewhat high generalization error. 
3. Approximation error - based on the decision boundary plot, the model does
not have a high approximation error, since it is capable or learning a z-shaped line that
is capable of separating half-moon structures well. 

"""

part1_q2 = r"""

Considering the data generating process, illustrated in the beginning of part 1, we see that there is an area ((x=0.5,x=1), (y=0.25,y=0.75)
that is densely populated by positive examples in the validation set, and by negative examples in the training set. Overall, observing
the two plots, we would expect more negative examples to be misclassified, i.e., higher FNR.

"""

part1_q3 = r"""
1. In this scenario we would prioritise not sending healthy patients to undergo the expensive, risky 
second examinations, namely we would want the FPR to be as low as possible. Hence, we would pick the
threshold corresponding to the lowest FPR on the ROC curve.  
2. In this scenario we would want the classifier to have high sensitivity, i.e. we would want
the TPR to be as high as possible, and therefore we would choose the threshold corresponding to the
highest TPR value on the ROC curve.

"""


part1_q4 = r"""
**Your answer:**
1. For a fixed depth, we can expect to achieve better(or at least not worse) accuracies the wider the network is,
since a narrow network can theoretically be contained in a wider one(by not using the extra neurons 
in each layer). However, optimization problems come into play and may prevent it, as we can see in the case
of depth=4 and width 128, which achieves a slightly lower accuracy than the w=32 network. The most significant
improvement is observed for depth=2 and width=32, compared to width=8. For depth=1, we notice a very small
improvement which we can attribute to the limitedness of a 1-layer deep network, which is probably not potent enough
to fit our data, no matter how wide the network is. 
2. For a fixed width, we observe better performance for deeper networks (apart from some exceptions
which probably arise from optimization errors). We notice that for width 2 and 4, increasing the depth from
1 to 2 is not enough to improve the accuracy significantly, which we relate to the overall low number
of parameters. In this case, increasing the depth from 2 to 4 does the trick. However, for width 8 and 32,
increasing the depth from 1 to 2 achieves the desired improvement while increasing further does not 
yield any further (significant) improvement.
3. We see that for the same number of parameters, a depth 4 is superior to depth 1 networks. This can be expected,
as 2 hidden layers (with sufficient width) are able to fit any arbitrary function, while 1 hidden layer isn't.
4. Choosing a threshold based on the validation set did improve the results on the test set compared to
the original accuracy of the validation set with the initial threshold. We can attribute it to the fact that the learning algorithm
and the validation data generalize well to the underlying distribution.
"""
# ==============
# Part 2 answers


def part2_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you return needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.1
    weight_decay = 0.01
    momentum = 0.3
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers


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

part3_q5 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
