import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        n_classes = x_scores.shape[1]
        #print(y.shape)
        one_hot = torch.logical_not(y.reshape(y.shape[0], 1) - torch.Tensor(range(n_classes)).reshape(1,n_classes)).float()
        R = torch.mm(x_scores, torch.transpose(one_hot, 0,1)).diagonal()
        M = torch.transpose(x_scores, 0, 1) - R
        #M = torch.max(M+torch.tensor([self.delta]), torch.tensor([0]))
        M = torch.relu(M+torch.tensor([self.delta]))
        loss = torch.sum(M)/M.shape[1] - self.delta
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx["M"] = torch.transpose(M, 0, 1)
        self.grad_ctx["one_hot"] = one_hot
        self.grad_ctx["X"] = x
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        # transform M to a 0,1 (0 remains 0, positive values become 1, simply
        # divide M by itself element-wise. EXCEPT that for w_j, all entries that correspond
        # to samples that were classified as j contain the minus sum of the rest of the row.
        # we have one_hot for assistance.

        one_hot = self.grad_ctx["one_hot"]
        M = self.grad_ctx["M"]
        X = self.grad_ctx["X"]
        #_M = M/M
        _M = (M>0).float()
        _M_sum_rows = -torch.sum(_M, dim=1) + self.delta
        _M_sum_rows = torch.diag(_M_sum_rows)

        one_hot_sums = torch.mm(_M_sum_rows, one_hot)

        G = _M - one_hot + one_hot_sums
        grad = torch.mm(torch.transpose(X, 0, 1), G) / X.shape[0]

        # ========================

        return grad
