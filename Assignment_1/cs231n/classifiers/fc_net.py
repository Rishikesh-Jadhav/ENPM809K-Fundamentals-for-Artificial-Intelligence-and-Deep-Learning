from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Neural Network Initialization Steps:
        # 1. Initialize the weights and biases for a two-layer neural network
        # 2. Scale the weights by 'weight_scale' and initialize them with random values
        #    drawn from a standard normal distribution
        # 3. Set the biases to zero
        # 4. Define parameters for both the forward and backward passes 
       
        # Initialize the weights and biases for the input-to-hidden layer 
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        
        # Initialize the weights and biases for the hidden-to-output layer
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Forward Pass Steps:
        # 1. Perform an affine transformation followed by the ReLU activation
        # 2. Apply another affine transformation
        # 3. Store cached values and outcomes for later use in the backward pass during training

        # Perform an affine transformation followed by the ReLU activation
        hidden_layer_output, hidden_layer_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])

        # Apply another affine transformation
        scores, output_layer_cache = affine_forward(hidden_layer_output, self.params['W2'], self.params['b2'])

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward Pass Steps:
        # 1. Compute the gradient of the loss with respect to the scores and apply softmax loss
        # 2. Add a regularization term to the loss to prevent overfitting
        # 3. Calculate gradients and perform the backward pass for the second layer
        # 4. Add a regularization term to the gradient of the weights for the second layer
        # 5. Calculate gradients and perform the backward pass for the first layer
        # 6. Add a regularization term to the gradient of the weights for the first layer

        # Compute the gradient of the loss with respect to the scores and apply softmax loss
        loss, dout = softmax_loss(scores, y)

        # Add a regularization term to the loss to prevent overfitting
        loss += 0.5 * self.reg * (np.sum(np.power(self.params['W1'], 2)) + np.sum(np.power(self.params['W2'], 2)))

        # Calculate gradients and perform the backward pass for the second layer
        dz, grads['W2'], grads['b2'] = affine_backward(dout, output_layer_cache)

        # Add a regularization term to the gradient of the weights for the second layer
        grads['W2'] += self.reg * self.params['W2']

        # Calculate gradients and perform the backward pass for the first layer
        dx, grads['W1'], grads['b1'] = affine_relu_backward(dz, hidden_layer_cache)

        # Add a regularization term to the gradient of the weights for the first layer
        grads['W1'] += self.reg * self.params['W1']


        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
