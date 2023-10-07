from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Adding the regularization loss to the gradient
    dW += 2 * reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Step 1: Compute Scores and Margins
    # We'll start by computing the scores for each class for all training examples. This involves multiplying the input data X by the weight matrix W. 
    # Then, we'll calculate the margins, which are the differences between the scores of the correct class and the scores of the other classes plus a margin of 1. 
    # We'll use broadcasting to efficiently compute these margins.
    
    num_train = X.shape[0]

    # Compute scores
    scores = X.dot(W)
    # Get the scores for the correct classes
    correct_scores = scores[np.arange(num_train), y]
    # Calculate margins (hinge loss)
    margins = scores - correct_scores[:, np.newaxis] + 1.0
    # Set margins of the correct class to 0
    margins[np.arange(num_train), y] = 0
    # Ensure margins are non-negative
    margins = np.maximum(0, margins)

    # Step 2: Compute the loss
    loss = np.sum(margins) / num_train

    # Add regularization to the loss
    loss += reg * np.sum(W * W)
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Step 3: Compute the Gradient
    # To compute the gradient of the loss with respect to the weights W, we have to calculate how much each weight contributes to the loss for each training example. 
    # We then haev to average contributions over all training examples.

    # Initialize binary matrix to show which margins are greater than 0
    binary_margins = (margins > 0).astype(int)
    # Calculate the sum of positive margins for each training example
    sum_positive_margins = np.sum(binary_margins, axis=1)
    # Subtract the number of positive margins for the correct class from the binary matrix 
    binary_margins[np.arange(X.shape[0]), y] = -sum_positive_margins
    # Compute gradientS
    dW = X.T.dot(binary_margins)
    # Average over the number of training examples and add the regularization gradient
    dW /= num_train
    # Add regularization loss to the gradient
    dW += 2 * reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
