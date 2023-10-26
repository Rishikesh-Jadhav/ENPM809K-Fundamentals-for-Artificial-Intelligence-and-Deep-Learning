from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Calculate the forward pass by applying weights and biases to the input data
    # Reshape input x and compute the dot product with weights w, then add biases b
    dim_size = x[0].shape
    out = x.reshape(x.shape[0], np.prod(dim_size)).dot(w) + b.reshape(1, -1)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Calculate gradients for the input data x, weights w, and biases b
    dim_shape = np.prod(x[0].shape)
    N = x.shape[0]
    X = x.reshape(N, dim_shape)

    # Compute gradients based on upstream gradient (dout)
    dx = dout.dot(w.T)  # (N x M) x (M x D) = (N x D)
    dx = dx.reshape(x.shape)

    dw = X.T.dot(dout)  # (D x N) x (N x M) = (D x M)

    db = dout.sum(axis=0)  # (N x M), so sum over all N

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Apply ReLU activation function element-wise to the input x
    out = np.maximum(0, x)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Calculate the gradient with respect to x based on the upstream gradient (dout)
    # ReLU acts as a switch, letting gradients pass through if they are > 0, else zeroing them out
    dx = dout
    dx[x < 0] = 0

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Shift the logits for numerical stability
    shifted_logits = x - np.max(x, axis=1, keepdims=True)

    # Calculate the denominator for the softmax function (Z)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)

    # Compute the log probabilities and softmax probabilities
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)

    N = x.shape[0]

    # Calculate the cross-entropy loss and gradients
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    layernorm = bn_param.get("layernorm", 0)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Step 1: Compute the sample mean from mini-batch statistics
        sample_mean = x.mean(axis=0)  # Shape (D,)

        # Step 2: Subtract the mean from the input
        x_minus_mean = x - sample_mean  # Shape (N, D)

        # Step 3: Compute the squared deviations
        squared_deviations = x_minus_mean ** 2  # Shape (N, D)

        # Step 4: Compute the sample variance
        sample_variance = 1./ N * np.sum(squared_deviations, axis=0)  # Shape (D,)

        # Step 5: Compute the standard deviation from the variance with epsilon for numerical stability
        standard_deviation = np.sqrt(sample_variance + eps)  # Shape (D,)

        # Step 6: Compute the inverted standard deviation
        inverted_stddev = 1./ standard_deviation  # Shape (D,)

        # Step 7: Normalize the input using the inverted standard deviation
        x_normalized = x_minus_mean * inverted_stddev  # Shape (N, D)

        # Step 8: Scale the normalized data using the scale parameter gamma
        scaled_x = gamma * x_normalized  # Shape (N, D)

        # Step 9: Shift the scaled data by the shift parameter beta
        out = scaled_x + beta  # Shape (N, D)

        # Store values for the backward pass
        cache = {'mean': sample_mean, 'stddev': standard_deviation, 'var': sample_variance, 'gamma': gamma, 
                 'beta': beta, 'eps': eps, 'x_norm': x_normalized, 'dev_from_mean': x_minus_mean,
                 'inverted_stddev': inverted_stddev, 'x': x}

        # Determine the axis for later backpropagation (1 for layernorm, 0 for batchnorm)
        cache['axis'] = 1 if layernorm else 0

        # Update the running mean and running variance (not applicable for layernorm)
        if not layernorm:
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_variance

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Normalize the input using the running mean and running variance
        z = (x - running_mean) / np.sqrt(running_var + eps)

        # Scale and shift the normalized data
        out = gamma * z + beta

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache

    
def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
 

    # Extract all relevant parameters from the cache
    beta, gamma, x_norm, var, eps, stddev, dev_from_mean, inverted_stddev, x, mean, axis = \
        cache['beta'], cache['gamma'], cache['x_norm'], cache['var'], cache['eps'], \
        cache['stddev'], cache['dev_from_mean'], cache['inverted_stddev'], cache['x'], \
        cache['mean'], cache['axis']

    # Obtain the number of training examples and the dimensionality of the input (number of features)
    N, D = dout.shape  # Alternatively, we can use x.shape

    # Gradient computation starts from the final step (backward pass)

    # Step 1: Gradient of beta is the sum of upstream derivatives along the specified axis
    dbeta = np.sum(dout, axis=axis)
    dscaled_x = dout  # No further derivative

    # Step 2: Gradient of gamma is the dot product of x_norm and dscaled_x along the specified axis
    dgamma = np.sum(x_norm * dscaled_x, axis=axis)
    dx_norm = gamma * dscaled_x

    # Step 3: Gradient of inverted standard deviation
    dinverted_stddev = np.sum(dev_from_mean * dx_norm, axis=0)
    ddev_from_mean = inverted_stddev * dx_norm

    # Step 4: Gradient of standard deviation
    dstddev = -1 / (stddev ** 2) * dinverted_stddev

    # Step 5: Gradient of variance
    dvar = (0.5) * 1 / np.sqrt(var + eps) * dstddev

    # Step 6: Gradient of squared deviations
    ddev_from_mean_sq = 1 / N * np.ones((N, D)) * dvar  # Variance of mean is 1/N

    # Step 7: Gradient of deviations from the mean
    ddev_from_mean += 2 * dev_from_mean * ddev_from_mean_sq

    # Step 8: Gradient of x
    dx = 1 * ddev_from_mean
    dmean = -1 * np.sum(ddev_from_mean, axis=0)

    # Step 9: Gradient of x, including additional contributions
    dx += 1. / N * np.ones((N, D)) * dmean

    # End of gradient computation

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta



def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Batchnorm_backward_alt is computed by simplifying the gradients derived on paper.
    # The sequence of derivatives follows the order in which they are calculated in the paper.
    # The convention used here is that downstream gradient equals local gradient times upstream gradient.


    # Extract all relevant parameters from the cache
    beta, gamma, x_norm, var, eps, stddev, dev_from_mean, inverted_stddev, mean, x, axis = \
        cache['beta'], cache['gamma'], cache['x_norm'], cache['var'], cache['eps'], \
        cache['stddev'], cache['dev_from_mean'], cache['inverted_stddev'], cache['mean'], \
        cache['x'], cache['axis']

    # Obtain the number of training examples and dimensionality of the input (number of features)
    N = dout.shape[0]  # Alternatively, we can use x.shape

    # Step 9: Gradient of beta is the sum of upstream derivatives along the specified axis
    dbeta = np.sum(dout, axis=axis)
    dscaled_x = dout  # No further derivative

    # Step 8: Gradient of gamma
    dgamma = np.sum((x - mean) * (var + eps)**(-1. / 2.) * dout, axis=axis)

    # Step 7: Gradient of mean
    dmean = 1 / N * np.sum(dout, axis=0)

    # Step 6: Gradient of variance
    dvar = 2 / N * np.sum(dev_from_mean * dout, axis=0)

    # Step 5: Gradient of standard deviation
    dstddev = dvar / (2 * stddev)

    # Step 4: Gradient of squared deviations from the mean
    dx = gamma * ((dout - dmean) * stddev - dstddev * dev_from_mean) / stddev**2

    # End of gradient computation

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # In layer normalization, all the hidden units in a layer share the same normalization
    # terms (mean and variance), but different training cases have different normalization terms.
    # Unlike batch normalization, layer normalization does not impose any constraint on the size
    # of a mini-batch and can be used in the pure online regime with any batch size.

    ln_param['mode'] = 'train'  # Set the mode to 'train', same as batch normalization in train mode

    # The forward pass here is similar to batch normalization. To maintain consistency and 
    # easily adapt batch normalization code, we set 'layernorm' in the ln_param dictionary.
    ln_param['layernorm'] = 1

    # Transpose x, gamma, and beta to match the batch normalization format
    out, cache = batchnorm_forward(x.T, gamma.reshape(-1, 1), beta.reshape(-1, 1), ln_param)

    # Transpose the output to restore the original dimensions (N, D)
    out = out.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Transpose dout because we transposed the input, x, during the forward pass
    dx, dgamma, dbeta = batchnorm_backward(dout.T, cache)

    # Transpose gradients w.r.t. input, x, to their original dimensions (N, D)
    dx = dx.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


