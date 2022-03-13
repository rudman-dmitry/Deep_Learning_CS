r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""
import torch

# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(
        batch_size=32, gamma=0.99, beta=0.05, learn_rate=0.0015, eps=1e-7, num_workers=0,
        hidden_dims=512
    )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======

    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(
        batch_size=32,
        gamma=0.99,
        beta=0.01,
        delta=0.01,
        learn_rate=0.002,
        eps=1e-7,
        num_workers=0,
        hidden_dims=512
    )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======

    # ========================
    return hp


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


part1_q3 = r"""
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

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(batch_size=32,
                  h_dim=512,
                  z_dim=128,
                  x_sigma2=0.0001,
                  learn_rate=0.001,
                  betas=(0.5, 0.55),
    ) 
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

$\sigma^2$ - likelihood variance, it does the regularization of the  data-reconstruction loss, which is the term in the total loss equation. High values will reduce the influence of this term on the total loss, this will favour the regularisation term over the reconstruction term, it will cause the images to be closer to the input. The opposite stands if $\sigma^2$ is low.


"""

part2_q2 = r"""
**Your answer:**

1. The KL divergence between two probability distributions simply measures how much they diverge from each other. Minimizing the KL divergence here means optimizing the probability distribution parameters to closely resemble that of the target distribution. The reconstruction term will try to improve the quality
of the reconstruction, neglecting the shape of the latent space.


2. KL divergence normalizes and makes the latent space smoother, reduces the overfitting to the training data.


3. If the parametrs of the loss are properly tuned - decoder will not just decode single, specific encodings in the latent space, but ones that slightly vary too, as the decoder is exposed to a range of variations of the encoding of the same input during training.

"""

part2_q3 = r"""
**Your answer:**

By maximizing the evidence distribution we provide our model an ability to generate data from the latent space with the same distribution as the data in an instance space.

"""

part2_q4 = r"""
**Your answer:**

We model the log to increase the range of the latent space distribution, since the values of the variance are always positive.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You can add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You can add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=4,
        z_dim=512,
        data_label=0,
        label_noise=0.4,
        discriminator_optimizer=dict(
            type="SGD",  # Any name in nn.optim like SGD, Adam
            lr=0.01,
            # You can add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="SGD",  # Any name in nn.optim like SGD, Adam
            lr=0.01,
            # You can add extra args for the optimizer here
        ))
    # ========================
    return hypers


part3_q1 = r"""
During training, we sample from the generator on two occasions:
1. When training the discriminator, we sample images from the generator and evaluate the discriminator's performance based upon its classification on them. Here, we do not take gradients of the samples since we're not trying to optimize the generator's loss function, we treat the generator's parameters as constant.
2. When training the generator, we wish to minimize its loss function. We obtain the current value of the loss function by sampling images from it and evaluating them with the discriminator. Here, we need to take gradients of the loss function w.r.t. the generator's parameters in order to optimise it w.r.t them. 
"""

part3_q2 = r"""
1. When training a GAN to generate images, should we decide to stop training solely based on the fact that the Generator loss is below some threshold? Why or why not?
The generator loss is being below some threshold does not necessarily mean that the generator is performing well. This is due to the fact that the generator loss is highly dependant on the discriminator's performance. Namely, if the discriminator does not perform well, we might get a low loss value for the generator, simply because the discriminator does not classify the samples well enough. Only a low loss value for the discriminator along with a low loss value for the generator may indicate that the generator is performing well.

2.What does it mean if the discriminator loss remains at a constant value while the generator loss decreases?
When the discriminator loss remains at a constant value while the generator loss decreases, we can infer the the generator is learning faster than the discriminator - that is, the discriminator is getting more powerful faster than the discriminator can keep up (their losses are tightly linked to one another).
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

# ==============


# ==============
# Part 4 answers
# ==============


def part4_affine_backward(ctx, grad_output):
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
