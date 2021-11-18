r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. **False.** The test set *does not* allow us to test our in-sample error. The in-sample error is by definition the
average of the loss function over the training set samples. The test set allows us to test the trained model and estimate its ability to generalize.
2. **False.** Different splits of the data into disjoint subsets may constitute training sets of different usefulness.
For instance, in an extreme case, if all positive samples appear before the negative ones in the dataset, the training set might consist
of only positive samples and bias the learning algorithm towards a positive classification. In other cases, the data might be
organized in other ways that yield non-descriptive or non-representative training set. For this reason, selecting the training set 
randomly from the data is more robust against artificial patterns.
3. **True.** The goal of cross-validation is to test the model's ability to predict new data that was not seen in training time.
The test set should not be included in the cross-validation process - using the test set during cross-validation may lead to
data leakage - the learning algorithm is at risk of overfitting the test set which will render the test set a faulty indicator of 
the learner ability to generalize. I.e., the chosen hyper-parameter values might be ones that overfit the test set and therefore achieve 
excellent results for it, but in reality do not generalize well for the underlying distribution.
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
Increasing k does not necessarily lead to better generalization for unseen data, as can be seen
in our cross-validation results. Indeed, if k is too low, the learning algorithm may be sensitive
to noise in the data and overfit it, and therefore not generalize well. In this sense increasing 
k up to a certain point leads to a better generalization capability. However, when k is too large, 
the algorithm underfits the data and fails to generalize well - it begins to treat all points in the
sampling space similarly, regardless of their location. At the extreme case where k=N, where 
N is the number of samples in the training set, the algorithm classifies all points in the sampling space
the same - as the mode class of the training set. 



"""

part2_q2 = r"""
**Your answer:**
1. Training on the entire train-set with different models and choosing the model with the best
accuracy on the train set will lead to an overfitting of the *train-set* and generalize poorly to unseen data. 
2. Training on the entire train-set with various models and selecting the model with the best *test accuracy* may overfit
 the *test-set* and not generalize well to unseen data.  
 
 
 In both cases, K-fold cross validation avoids the problem of overfitting by estimating the generalization error and choosing
  the models with the lowest estimated generalization error. The estimation of the generalization error is conducted on 
  different validation sets that weren't seen by the algorithm at training time, and averaging over them. This renders the
  estimation more accurate and less sensitive to noise in the training data.  
 


"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
The value of $\Delta>0$ determines how much larger the score of a sample's true class should be in comparison
to another class, in order to not be penalized.


I.e., $\Delta>0$ 's purpose is to ensure that a sample falls well "within"
its ground-truth label hyperplane in comparison to the other classes' hyperplanes. 
Different (numerically reasonable) values of $\Delta>0$ will yield equivalent optimization problems (equivalent loss functions)
that only differ by their scaling.

"""

part3_q2 = r"""
**Your answer:**
1. 
For each class, we expect the learning algorithm to find weights which give high scores to samples from the class 
(i.e., have a high inner-product with them) and low scores for samples from other classes (have a low inner-product with them).
A way to achieve that is to have high-values where there should be high values(to reward correct samples), and low values where there shouldn't be high values
(to penalize incorrect samples). That is why for a given entry(\pixel in the image)
and a given class $j$, we expect $w_j$ to have a high value in this entry if it is likely to be black in a sample from class *j*, and low otherwise.
This is supported by the visualization of the learned weights. For each digit - the weights are white where the digit should appear(where the entries in the input
are expected to be white) and darker elsewhere. That is, the linear algorithm learns which pixels should be white and which should be black by establishing
weights that are similar to the average input from each class.
We can further see that certain areas are
 particularly penalized - for example, the inner circle of the digit 0 - as opposed to others which can be both black or white.
 
 Some classification errors arise from class properties that are not fully captured by such a representation, or properties/features whose representation
  is too dependant upon location. For instance, we can see
  that a 3 was misclassified as a 2, possibly because the 3 is slightly tilted. That is, the representation of 3 expects the three edges of the
    "horizontal" parts to be lower and the middle one more inward. We also observe that the representations for 2 and 7 are rather similar,
     and a 2 was misclassified as a 7, probably because the lower horizontal line of the 2 was a bit higher than captured by the corresponding weight
      and was misclassified as a middle horizontal line of a 7. Here again we see over-sensitivity to the exact location of parts of digits, 
      and also insufficient sensitivity to other properties, such as the curly upper line of a 2 as opposed to the (usually) straight upper line of a 7.

2. Generally speaking, both models employ some kind of a similarity rule to determine which class an input sample belongs to. 
However, the linear model determines similarity to a class by creating a general "stencil" representation to it out of the input data, and uses the inner-product
 with it as the similarity score, 
whereas the knn model uses all training data as representatives and determines the similarity to each class (roughly speaking) by the euclidean distance to its
 representatives. More accurately, the knn model determines which class a given sample is most similar to by checking which class representatives it is the closest to. 

 

"""

part3_q3 = r"""
**Your answer:**
1. Based on the loss graph, the learning rate we chose is good, the loss declines exponentially and converges. Too low a learning rate would
 induce an overly slow learning, even when the loss function and its gradients assume high values. 
 In this case we'd observe a much slower decline, and the loss would not converge within the given epochs.
Conversely, if the learning rate was too high, we would notice a jagged decline, or no decline at all. Too big a stride in the opposite direction of the gradient
 would throw us off the direction of a local\global minimum and go uphill instead of downhill. Too high a learning rate would also most likely
  not converge whatsoever since the big strides will miss local\global minima.
  
2. We would say that our model neither overfits nor underfits the training set. We observe that the train and validation curves grow together to a point of stability.
If the model overfitted the training set the validation curve would begin to decrease from a certain point on.
On the other hand, if the model underfitted the data, the training curve would be on the increase at the end of the learning process.  
Overall we would say that the model constructs fairly simple (low resolution) representations for the classes that are not overly sensitive to noise\
 \arbitrary features in the dataset and thus do not overfit it, but their simplicity also prevents the model from fully capturing different classes' subtleties. 
 
"""

# ==============

# ==============
# Part 4 answers
part4_q1 = r"""
*Your answer:*


The ideal pattern in a residual plot will be represented by the normally distributed y-hats across the x-axis. The indication of how our model fits the data we will get from the proximity of values to the zero line.

"""

part4_q2 = r"""
*Your answer:*

1. The model is linear. By adding nonlinear features we increase the capacity of a model to fit the data, which may be not linear.

2. When there is a non-linear relation between the target and the features, fitting the right non-linear function will result in better linear separability in new coordinate space. Such transformations doesn't help to any nonlinearity, only the ones following the speciffic distribution

3. Adding non-linear features will not affect the decision boundary, it just will increase the separability.
"""

part4_q3 = r"""
*Your answer:*

1. We used np.logspace because values of 𝜆 are small, it increases their range.

2. N = k_folds * lambda_range * degree_range = 3 * 20 * 4 = 240


"""
# ==============


