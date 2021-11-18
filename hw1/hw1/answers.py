r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. **False.** Test set allows us to test the trained model and estimate it's ability of generalization.
2. **False.** The split is affected by the representativeness of train and test set.
3. **True.** The goal of cross-validation is to test the model's ability to predict new data that was not used in estimating it.
4. **True.** It's not a perfect indicator of model generalizability simply because we DO see the model performance on the validation set, and so we subconsciously tune hyperparameters to better fit the validation set accuracy (or in the case of k-fold cross-validation, validation set(s). A better evaluation of a models generalizability is on a "test" set.


"""

part1_q2 = r"""
**Your answer:** 

It's not a good approach. Now he just overfiiting regarding to the test set, he doesn't improve generalizabilty. It can result in minor improvement, but this approach is not justified.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
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

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


The ideal pattern in a residual plot will be represented by the normally distributed y-hats across the x-axis. The indication of how our model fits the data we will get from the proximity of values to the zero line.

"""

part4_q2 = r"""
**Your answer:**

1. The model is linear. By adding nonlinear features we increase the capacity of a model to fit the data, which may be not linear.

2. When there is a non-linear relation between the target and the features, fitting the right non-linear function will result in better linear separability in new coordinate space. Such transformations doesn't help to any nonlinearity, only the ones following the speciffic distribution

3. Adding non-linear features will not affect the decision boundary, it just will increase the separability.
"""

part4_q3 = r"""
**Your answer:**

1. We used np.logspace because values of 𝜆 are small, it increases their range.

2. N = k_folds * lambda_range * degree_range = 3 * 20 * 4 = 240
 

"""

# ==============
