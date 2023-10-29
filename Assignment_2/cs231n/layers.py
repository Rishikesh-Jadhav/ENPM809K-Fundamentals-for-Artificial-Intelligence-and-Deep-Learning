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

    # Transpose dout because we transposed the input, x, during the forward pass
    dx, dgamma, dbeta = batchnorm_backward(dout.T, cache)

    # Transpose gradients w.r.t. input, x, to their original dimensions (N, D)
    dx = dx.T

    return dx, dgamma, dbeta


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 

    Returns a tuple of:
    - output_data: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)

    HH-filter height
    WW-filter width
    """
    output_data = None

    # Extract parameters
    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    # Check if the padding and stride settings are valid
    assert (H + 2 * pad - HH) % stride == 0, '[Sanity Check] [FAIL]: Conv Layer Failed in Height'
    assert (W + 2 * pad - WW) % stride == 0, '[Sanity Check] [FAIL]: Conv Layer Failed in Width'

    # Calculate the height of the output feature maps based on the provided padding, stride, and input dimensions.
    output_height = (H + 2 * pad - HH) // stride + 1 
    # Calculate the width of the output feature maps
    output_width  = (W + 2 * pad - WW) // stride + 1

    # Create an output volume tensor after convolution(Initialze the output data as a 4d numpy array)
    output_data = np.zeros((N, F, output_height, output_width ))

    # Pad height and width axes of the input data with zeros 
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    # Naive Loops: Perform convolution
    for n in range(N):  #  Iterate over each data point in the input
        for f in range(F): #  Iterate over each filter or kernel
            for i in range(0, output_height): #  Iterate over each vertical activation position in the output feature maps
                for j in range(0, output_width ): #  Iterate over each horizontal activation position in the output feature maps
                    # Calculate the convolution result for each neuron
                    # Multiply the corresponding regions of the input data and filter weights, sums the results, and adds the bias term to obtain the output value for that specific neuron
                    output_data[n, f, i, j] = (x_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] * w[f, :, :, :]).sum() + b[f]
                    #out                    =                           X                                W                  +   b

    # Store the input data, filter weights, biases, and convolution parameters for later use
    cache = (x, w, b, conv_param)
    return output_data, cache

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives. parameter represents the gradients coming from the layer above in the network
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive. tuple containing cached values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    # Initialize variables to store the gradients
    dx, dw, db = None, None, None

    # Extract parameters from the cache
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape # w is filter weights
    stride = conv_param.get('stride', 1) # get stride value else set to 1
    pad = conv_param.get('pad', 0)

    # Apply zero-padding to the input data, creating a new array x_pad. Padding ensures that the spatial dimensions are preserved during convolution
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    # Calculate the height of the output feature maps based on the input dimensions, filter size, padding, and stride
    output_height  = (H + 2 * pad - HH) // stride + 1 
    # Calculate the width of the output feature maps 
    output_width  = (W + 2 * pad - WW) // stride + 1

    # Initialize output gradients to store gradients wrt input data, weights, biases and wrt padded input
    dx_pad = np.zeros_like(x_pad)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # Naive Loops: Compute gradients
    for n in range(N): # Iterate over each data point in the input
        for f in range(F): # Iterate over each filter or kernel.
            db[f] += dout[n, f].sum() # Accumulate the gradients for the biases by summing the upstream gradients (dout) for the corresponding neuron and filter
            for i in range(0, output_height): # For each y activation
                for j in range(0, output_width): # For each x activation
                    # Calculate the gradients for the filter weights by multiplying the corresponding regions of the input data and upstream gradients and accumulating the results
                    dw[f] += x_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] * dout[n, f, i, j]      
                    #  Update dx_pad by accumulating the product of the filter weights and upstream gradients, adjusted for the appropriate spatial positions
                    dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += w[f] * dout[n, f, i, j]
    
    # Extract input_data_gradients (dx) from input_data_gradients_padded since dx.shape should match x.shape
    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]

    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None

    # Extract parameters
    N, C, H, W = x.shape
    HH = pool_param.get('pool_height', 2)
    WW = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)

    # Check if the input dimensions and pooling parameters are valid
    assert (H - HH) % stride == 0, '[Sanity Check] [FAIL]: Conv Layer Failed in Height'
    assert (W - WW) % stride == 0, '[Sanity Check] [FAIL]: Conv Layer Failed in Width'

    # Calculate the output feature map size
    output_height = (H - HH) // stride + 1
    output_width = (W - WW) // stride + 1

    # Create the output feature map tensor after max-pooling
    out = np.zeros((N, C, output_height, output_width)) # output has same dims NCHW format as input

    # Naive loops to perform max-pooling
    for n in range(N): # For each data point
        for i in range(output_height): # For each y activation
            for j in range(output_width): # For each x activation
                # Apply max-pooling by taking the maximum value within each pooling region
                out[n, :, i, j] = np.amax(x[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW], axis=(-1, -2))
    
    # Cache the input data and pooling parameters for later use during backward pass.
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    # Extract constants and shapes,including the input dimensions and pooling parameters.
    x, pool_param = cache
    N, C, H, W = x.shape
    HH = pool_param.get('pool_height', 2)
    WW = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)

    # Calculate the dimensions of the output feature maps based on the input dimensions and pooling parameters
    output_height = (H - HH) // stride + 1 
    output_width = (W - WW) // stride + 1

    # Initialized with zeros to store the gradient with respect to the input data
    dx = np.zeros_like(x)
    
    # Naive loops to compute the gradients
    for n in range(N): # For each data point
        for c in range(C): # For each channel
            for i in range(output_height): # For each y activation
                for j in range(output_width): # For each x activation
                    # Determine the indices of the maximum value in the corresponding pooling region
                    ind = np.argmax(x[n, c, i*stride:i*stride+HH, j*stride:j*stride+WW])
                    # Convert the flattened index into a 2D index within the pooling region
                    ind1, ind2 = np.unravel_index(ind, (HH, WW))
                    
                    # Pass the gradient only through the index of the maximum value                    
                    dx[n, c, i*stride:i*stride+HH, j*stride:j*stride+WW][ind1, ind2] = dout[n, c, i, j]

    return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    # Get the dimensions of the input data
    N, C, H, W = x.shape

    # Transpose the input to a channel-last notation (N, H, W, C)
    # and then reshape it to normalize over N*H*W for each channel (C)
    x = x.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    # Call the batch normalization forward pass on the reshaped data
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)

    # Transpose the output back to the original shape (N, C, H, W)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    # Initialize gradients with respect to input (dx), scale parameter (dgamma), and shift parameter (dbeta)
    dx, dgamma, dbeta = None, None, None

    N, C, H, W = dout.shape

    # Transpose the input to a channel-last notation (N, H, W, C)
    # and then reshape it to normalize over N*H*W for each channel (C)
    dout = dout.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    # Calculate gradients using the batchnorm_backward_alt function
    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)

    # Transpose the gradient output back to its original shape (N, C, H, W)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)    

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)

    # key idea of Groupnorm: compute mean and variance statistics by dividing 
    # each datapoint into G groups 
    # gamma/beta (shift/scale) are per channel

    # Extract parameters and parameters for the group normalization
    N, C, H, W = x.shape
    size = (N*G, C//G * H * W) # in groupnorm, D = C//G * H * W

    # Step 0 - Reshape input data to accommodate groups (G)
    # divide each sample into G groups (G new samples)
    x = x.reshape((N*G, -1)) # reshape to same as size # reshape NxCxHxW ==> N*GxC/GxHxW =N1*C1 (N1>N*Groups)

    # Step 1 - Compute group mean by averaging over each group
    # mini-batch mean by averaging over a particular column / feature dimension (D)
    # over each sample (N) in a minibatch 
    mean = x.mean(axis = 1, keepdims= True) # (N,1) # sum through D
    # can also do mean = 1./N * np.sum(x, axis = 1)

    # Step 2 - Subtract group mean from the data
    dev_from_mean = x - mean # (N,D)

    # Step 3 - Compute the squared deviation from the mean
    dev_from_mean_sq = dev_from_mean ** 2 # (N,D)

    # Step 4 - Compute group variance by averaging squared deviations
    var = 1./size[1] * np.sum(dev_from_mean_sq, axis = 1, keepdims= True) # (N,1)
    # can also do var = x.var(axis = 0)

    # Step 5 - Compute the standard deviation from variance, add eps for numerical stability
    stddev = np.sqrt(var + eps) # (N,1)

    # Step 6 -Invert the standard deviation to obtain the denominator
    inverted_stddev = 1./stddev # (N,1)

    # Step 7 - Apply normalization using the inverted standard deviation
    x_norm = dev_from_mean * inverted_stddev # also called z or x_hat (N,D) 
    x_norm = x_norm.reshape(N, C, H, W)

    # Step 8 - Apply scaling parameter (gamma) to normalized data
    scaled_x = gamma * x_norm # (N,D)

    # Step 9 - Shift the scaled data by the shift parameter (beta)
    out = scaled_x + beta # (N,D)

    # Backpropagation variables
    axis = (0, 2, 3)

    # Cache values for backward pass
    cache = {'mean': mean, 'stddev': stddev, 'var': var, 'gamma': gamma, \
             'beta': beta, 'eps': eps, 'x_norm': x_norm, 'dev_from_mean': dev_from_mean, \
             'inverted_stddev': inverted_stddev, 'x': x, 'axis': axis, 'size': size, 'G': G, 'scaled_x': scaled_x}
    
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    # Initialize gradients for input (dx), scale parameter (dgamma), and shift parameter (dbeta)
    dx, dgamma, dbeta = None, None, None

    # Convention used is downstream gradient = local gradient * upstream gradient
    # Extract all relevant parameters and intermediate values from the cache
    beta, gamma, x_norm, var, eps, stddev, dev_from_mean, inverted_stddev, x, mean, axis, size, G, scaled_x = \
    cache['beta'], cache['gamma'], cache['x_norm'], cache['var'], cache['eps'], \
    cache['stddev'], cache['dev_from_mean'], cache['inverted_stddev'], cache['x'], \
    cache['mean'], cache['axis'], cache['size'], cache['G'], cache['scaled_x']

    N, C, H, W = dout.shape
    
    # (9) Calculate the gradient with respect to shift parameter (dbeta)
    dbeta = np.sum(dout, axis = (0,2,3), keepdims = True) #1xCx1x1
    dscaled_x = dout # N1xC1xH1xW1

    # (8) Calculate the gradient with respect to scale parameter (dgamma)
    dgamma = np.sum(dscaled_x * x_norm,axis = (0,2,3), keepdims = True) # N = sum_through_D,W,H([N1xC1xH1xW1]xN1xC1xH1xW1)
    dx_norm = dscaled_x * gamma # N1xC1xH1xW1 = [N1xC1xH1xW1] x[1xC1x1x1]
    dx_norm = dx_norm.reshape(size) #(N1*G,C1//G*H1*W1)

    # (7) Calculate the gradient with respect to inverted standard deviation (dinverted_stddev)
    dinverted_stddev = np.sum(dx_norm * dev_from_mean, axis = 1, keepdims = True) # N = sum_through_D([NxD].*[NxD]) =4×60
    ddev_from_mean = dx_norm * inverted_stddev #[NxD] = [NxD] x [Nx1]

    # (6) Calculate the gradient with respect to standard deviation (dstddev)
    dstddev = (-1/(stddev**2)) * dinverted_stddev # N = N x [N]

    # (5) Calculate the gradient with respect to variance (dvar)
    dvar = 0.5 * (1/np.sqrt(var + eps)) * dstddev # N = [N+const]xN

    # (4) Calculate the gradient with respect to squared deviation from the mean (ddev_from_mean_sq)
    ddev_from_mean_sq = (1/size[1]) * np.ones(size) * dvar # NxD = NxD*N

    # (3) Calculate the gradient with respect to the deviation from the mean (ddev_from_mean)
    ddev_from_mean += 2 * dev_from_mean * ddev_from_mean_sq # [NxD] = [NxD]*[NxD]

    # (2) Calculate the gradient with respect to the normalized data (dx_norm)
    dx = (1) * ddev_from_mean # [NxD] = [NxD]
    dmean = -1 * np.sum(ddev_from_mean, axis = 1, keepdims = True) # N = sum_through_D[NxD]

    # (1) Calculate the gradient with respect to the group mean (dmean)
    dx += (1/size[1]) * np.ones(size) * dmean # NxD (N= N1*Groups) += [NxD]XN

    # (0) Calculate the gradient with respect to the inputs (dx) and reshape it to the original shape
    dx = dx.reshape(N, C, H, W)

    return dx, dgamma, dbeta

