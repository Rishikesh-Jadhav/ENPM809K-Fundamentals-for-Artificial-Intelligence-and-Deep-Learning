from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
      
        # Initialize network parameters
        # Weights are initialized from a normal distribution, biases are initialized to zero
        # For batch/layer normalization, scale parameters (gamma) are initialized to ones, and shift parameters (beta) are initialized to zeros
     
        layer_dims = np.hstack([input_dim, hidden_dims, num_classes])

        # Loop through each layer, from the input layer to the output layer
        for layer in range(self.num_layers):
            # Initialize weights ('W') using a normal distribution with the specified 'weight_scale'
            # The dimensions of 'W' depend on the current layer and the next layer  
            self.params['W' + str(layer + 1)] = np.random.randn(layer_dims[layer], \
                                                layer_dims[layer + 1]) * weight_scale
            # Initialize biases ('b') to zeros
            # The dimensions of 'b' depend on the size of the current layer
            self.params['b' + str(layer + 1)] = np.zeros(layer_dims[layer + 1])

        # If batch normalization is enabled, we initialize scale ('gamma') to ones and shift ('beta') to zeros for each layer.
        if self.normalization:
            # Loop through all but the output layer
            for layer in range(self.num_layers - 1):
                # Initialize scale ('gamma') to ones
                self.params['gamma' + str(layer + 1)] = np.ones((layer_dims[layer + 1]))
                # Initialize shift ('beta') to zeros        
                self.params['beta' + str(layer + 1)] = np.zeros((layer_dims[layer + 1]))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Let's perform the forward pass through the network.
        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

        caches = {} # Create a dictionary to store intermediate results and caches for each layer

        # Loop through (L - 1) layers with ReLU activation
        for layer in range(self.num_layers - 1):
            W = self.params['W' + str(layer + 1)]# Get the weights for the current layer
            b = self.params['b' + str(layer + 1)]# Get the biases for the current layer
            
            if self.normalization:
                gamma = self.params['gamma' + str(layer + 1)] # Get gamma (scale) parameters for normalization
                beta  = self.params['beta' + str(layer + 1)]  # Get beta (shift) parameters for normalization

                bn_params = self.bn_params[layer]   # Get batch/layer normalization parameters
            
                # Perform the affine - ReLU forward pass without batch/layer normalization
                X, cache = affine_norm_relu_forward(X, W, b, gamma, beta, bn_params, \
                                                    self.normalization)
            else:
                X, cache = affine_relu_forward(X, W, b)
            # Store the computed results and cache for the current layer
            caches[layer + 1] = cache

            # Apply dropout after ReLU if 'self.use_dropout' is enabled
            if self.use_dropout:
                X, cache = dropout_forward(X, self.dropout_param)                
                caches['dropout' + str(layer + 1)] = cache

        # (2) Perform the forward pass for the output layer: affine - softmax
        W = self.params['W' + str(self.num_layers)]
        b = self.params['b' + str(self.num_layers)]
        
        # Compute the final class scores (unnormalized probabilities) for classification
        scores, cache = affine_forward(X, W, b)
        # Store the computed results and cache for the output layer
        caches[self.num_layers] = cache   

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Let's perform the backward pass through the network.
        # (2) Compute the data loss using softmax and obtain the initial gradient 'dout'
        loss, dout = softmax_loss(scores, y)

        # Add the regularization loss term to the overall loss
        for layer in range(self.num_layers):
            W = self.params['W' + str(layer + 1)]
            loss += 0.5 * self.reg * np.sum(W * W)         

        # (2) (1) {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine
        # Split the computation for the last layer since it doesn't use ReLU and doesn't perform batch/layer normalization. The first (L - 1) layers include ReLU activations and potential dropout.
        # Handle the last hidden layer (no batch normalization, no ReLU, no dropout)
        dout, dw, db = affine_backward(dout, caches[self.num_layers])
        grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db

        # For the remaining layers, work backward from the last hidden layer to the input layer
        for layer in range(self.num_layers - 2, -1, -1):
            # If dropout was applied, perform backward dropout before reaching the last hidden layer
            if self.use_dropout:
                dout = dropout_backward(dout, caches['dropout' + str(layer + 1)])

            if self.normalization: # Check if normalization is enabled
                # Handle the backward pass for batch/layer normalization, ReLU, and affine operations
                dout, dw, db, dgamma, dbeta = affine_norm_relu_backward(dout, caches[layer + 1], \
                                                                      self.normalization)
                grads['gamma' + str(layer + 1)] = dgamma
                grads['beta' + str(layer + 1)] = dbeta
            else:
                # Handle the backward pass for ReLU and affine operations (no normalization)
                dout, dw, db = affine_relu_backward(dout, caches[layer + 1])

            # save data loss and add derivative of the regularization loss to dw
            grads['W' + str(layer + 1)] = dw + self.reg * self.params['W' + str(layer + 1)]
            grads['b' + str(layer + 1)] = db
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def affine_norm_relu_forward(x, w, b, gamma, beta, bn_params, normalization):
        """
        Convenience/"sandwich"/helper layer that combines multiple operations into commonly used patterns.
        Performs affine - batch/layer norm - relu.

        Inputs:
        - x: Input to the affine layer
        - w, b: Weights for the affine layer
        - gamma, beta: Batchnorm/Layernorm learnable params
        - bn_params: Batchnorm/Layernorm params
        - normalization: Are we using Batchnorm or Layernorm?

        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Tuple containing the cache of each layer to give to the backward pass
        
        """

        fc_cache, bn_cache, relu_cache = None, None, None

        # Affine layer: Computes the affine transformation.
        out, fc_cache = affine_forward(x, w, b)

        # ReLU activation: Applies element-wise rectified linear unit (ReLU) activation.
        out, relu_cache = relu_forward(out)

        return out, (fc_cache, bn_cache, relu_cache)

def affine_norm_relu_backward(dout, cache, normalization):
        """
        Backward pass for the affine - batch/layer norm - relu convenience layer.
        """
        
        fc_cache, bn_cache, relu_cache = cache

        # Backward ReLU: Computes the gradients of the loss with respect to ReLU's input.
        dout = relu_backward(dout, relu_cache)

        # Backward affine layer: Computes the gradients of the loss with respect to the input data, weights, and biases.
        dx, dw, db = affine_backward(dout, fc_cache)
        
        return dx, dw, db     
