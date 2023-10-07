from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # The following steps will help us to find the softmax loss and gradient using loops.
    # 1. Initialize the loss and gradient to zero.
    # 2. Calculate the scores (not normalized log probabilities) for each example and each class
    # 3. Loop over Training Examples and for each training example:
    #   - Subtract the maximum score in each row from all scores to avoid numerical instability.
    #   - Compute softmax probabilities for each class for the current example.
    #   - Add the negative log of the correct class probability to the loss.
    #   - Compute the gradient contributions for each class and feature, and subtract an additional contribution from the correct class.   
    # 4. Divide the loss and gradient by the number of training examples to obtain average values.
    # 5. Add the regularization term to the loss and add the gradient of the regularization term to the gradient
    # 6. Return the loss and gradient.

    # Computing the scores for each class for each example. This step essentially computes the linear scores before applying the softmax function.
    scores = X.dot(W)  
    no_train = X.shape[0]
    no_classes = W.shape[1]

    # We iterate over every training example in the minibatch
    for i in range(no_train):
      # Subtracting the max
      scores_new = scores[i] - np.max(scores[i]) 
      # Calculating the softmax probabilities for each class using the formula
      softmax = np.exp(scores_new)/np.sum(np.exp(scores_new))
      # Adding the negative log of the correct class probability to the loss
      loss += -np.log(softmax[y[i]])

      for j in range(no_classes):
        # Computing gradient contributions for each class and feature 
        dW[:,j] += X[i] * softmax[j]
      dW[:,y[i]] -= X[i]

    # Averaging the loss and gradient over the entire minibatch
    loss /= no_train
    dW /= no_train

    # Regularization helps prevent overfitting by penalizing large weights.
    # Adding the regularization term to the loss
    loss += reg * np.sum(W * W)
    # Adding the gradient of the regularization term to dW
    dW += reg * 2 * W 

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 1. Initialize the loss and gradient to zero.
    # 2. Determine the number of training examples in the input data.
    # 3. Calculate the scores (unnormalized log probabilities) for each class for each example 
    # 4. Subtract the maximum score in each row to prevent numerical instability when exponentiating.
    # 5. Softmax Loss Calculation
    #   - Calculate the exponentials of the shifted scores.
    #   - Compute the sum of exponentials for each example.
    #   - Calculate the softmax probabilities for each class for each example.
    #   - Calculate the negative log-likelihood of the correct class for each example and sum them to get the loss.
    # 6. Weight Gradient Calculation
    #   - Subtract 1 from the predicted probability of the correct class for each example.
    #   - Compute the gradient of the loss with respect to the weights.
    # 7. Divide the loss and gradient by the number of training examples to obtain average values.
    # 8. Add the regularization term to the loss and add the gradient of the regularization term to the gradient.

    # Geting the number of training examples
    no_train = X.shape[0]
    # Computing the raw scores (unnormalized log probabilities)
    scores = X.dot(W)
    scores -= scores.max(axis=1, keepdims=True)

    # Calculating the softmax probabilities
    softmax_probs  = np.exp(scores)/np.sum(np.exp(scores), axis = 1, keepdims = True)

    # Calculating the negative log likelihood of the correct class for each example and sum them to get the loss
    loss = np.sum(-np.log(softmax_probs [np.arange(no_train), y]) )

    # Computing the gradient of the loss with respect to the weights
    softmax_probs[np.arange(no_train),y] -= 1
    dW = X.T.dot(softmax_probs)

    # Averaging the loss and gradient over the entire minibatch
    loss /= no_train
    dW /= no_train

    # Adding the regularization term to the loss
    loss += reg * np.sum(W * W)
    # Adding the gradient of the regularization term
    dW += reg * 2 * W 

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
