"""
Source Code for Homework 3.b of ECBM E6040, Spring 2016, Columbia University

Instructor: Prof. Aurel A. Lazar

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import numpy

import theano
import math
import theano.tensor as T
from theano.tensor.signal import downsample

from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn, GaussianNetConvPoolLayer

def test_lenet_Gaussian(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512],
            batch_size=200, verbose=False):
    """
    Wrapper function for testing LeNet on SVHN dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # TODO: Construct the first convolutional pooling layer
    layer0 = GaussianNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0],3,5,5),
        poolsize=(2,2)
       # sigma = 2
    )

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 14, 14),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2,2)
       # sigma = 2
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer2_input = layer1.output.flatten(2)

    # TODO: construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 5 * 5,
        n_out=500,
        activation=T.tanh
    )

    # TODO: classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in= 500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # TODO: create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
    return layer0.W.get_value()


# TODO
"""
test_lenet(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512],
        batch_size=200,filtershape = 2,pool_size = 2,imageshape = 32,verbose=False)
"""
def test_lenet(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512], batch_size=200, filtershape = 2,pool_size=2,imageshape=32,verbose=False):
    """
    Wrapper function for testing LeNet on SVHN dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # TODO: Construct the first convolutional pooling layer
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0],3,filtershape,filtershape),
        poolsize=(pool_size,pool_size)
    )
    imageshape = ( (imageshape - filtershape + 1) / (pool_size ) )
    #imageshape[1] =  ( (imageshape[1] - filtershape[1] + 1) / (pool_size[1] ) )

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], imageshape, imageshape),
        filter_shape=(nkerns[1], nkerns[0], filtershape, filtershape),
        poolsize=(pool_size,pool_size)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer2_input = layer1.output.flatten(2)
    imageshape = ( (imageshape - filtershape + 1) / (pool_size ) )


    # TODO: construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * imageshape * imageshape,
        n_out=500,
        activation=T.tanh
    )

    # TODO: classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in= 500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # TODO: create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
    return layer0.W.get_value()

# TODO

def test_convnet(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512, 20],
        batch_size=200, verbose=False,filtershape = [2,2],pool_size = [2,2],imageshape = [32,32],downsizing = [2,2],ignore = True):
    """
    Wrapper function for testing Multi-Stage ConvNet on SVHN dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))
    

    # TODO: Construct the first convolutional pooling layer:
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape= (batch_size, 3, 32, 32),
        filter_shape= (nkerns[0], 3, filtershape[0], filtershape[1]),
        poolsize= (pool_size[0],pool_size[1]),
        pool_ignore_border=ignore
    )
    imageshape[0] =  int( math.ceil( float(imageshape[0] - filtershape[0] + 1) / float(pool_size[0] ) ) )
    imageshape[1] =  int( math.ceil( float(imageshape[1] - filtershape[1] + 1) / float(pool_size[1] ) ) )

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape= (batch_size, nkerns[0], imageshape[0], imageshape[1]),
        filter_shape= (nkerns[1], nkerns[0], filtershape[0], filtershape[1]),#TODO
        poolsize= (pool_size[0],pool_size[1]),#TODO
        pool_ignore_border=ignore
    )

    # Combine Layer 0 output and Layer 1 output
    # TODO: downsample the first layer output to match the size of the second
    # layer output.
    layer0_output_ds = downsample.max_pool_2d(
            input=layer0.output,
            ds= (downsizing[0],downsizing[1]), # TDOD: change ds
            ignore_border = ignore
    )
    # concatenate layer
    layer2_input = T.concatenate([layer1.output, layer0_output_ds], axis=1)
                     
                 

    # TODO: Construct the third convolutional pooling layer
    #imageshape[0] = ( (imageshape[0] - filtershape[0] + 1) / (pool_size[0] ) )
    #imageshape[1] =  ( (imageshape[1] - filtershape[1] + 1) / (pool_size[1] ) )
    imageshape[0] =  int( math.ceil( float(imageshape[0] - filtershape[0] + 1) / float(pool_size[0] ) ) )
    imageshape[1] =  int( math.ceil( float(imageshape[1] - filtershape[1] + 1) / float(pool_size[1] ) ) )

    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer2_input,
        image_shape= (batch_size, nkerns[0] + nkerns[1], imageshape[0], imageshape[1]),
        filter_shape= ( nkerns[2], nkerns[0] + nkerns[1], filtershape[0], filtershape[1]),
        poolsize= (pool_size[0],pool_size[1]),
        pool_ignore_border=ignore
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[2] * 1 * 1).
    layer3_input = layer2.output.flatten(2)
    #imageshape[0] = ( (imageshape[0] - filtershape[0] + 1) / (pool_size[0] ) )
    #imageshape[1] =  ( (imageshape[1] - filtershape[1] + 1) / (pool_size[1] ) )
    imageshape[0] =  int( math.ceil( float(imageshape[0] - filtershape[0] + 1) / float(pool_size[0] ) ) )
    imageshape[1] =  int( math.ceil( float(imageshape[1] - filtershape[1] + 1) / float(pool_size[1] ) ) )


    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * imageshape[0] * imageshape[1],
        n_out= 500,#TODO,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output,
        n_in= 500,#TODO
        n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # TODO: create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params#TODO

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
# TODO
def test_CDNN(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512, 20],
        batch_size=200, n_hiddenLayers = 3, n_hidden = 500,verbose=False):
    """
    Wrapper function for testing CNN in cascade with DNN
    """
    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))
    filtershape = [2,2]
    pool_size = [2,2]
    imageshape = [32,32]

    # TODO: Construct the first convolutional pooling layer:
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape= (batch_size, 3, 32, 32),
        filter_shape= (nkerns[0], 3, filtershape[0], filtershape[1]),
        poolsize= (2,2)
    )
    imageshape[0] = ( (imageshape[0] - filtershape[0] + 1) / (pool_size[0] ) )
    imageshape[1] =  ( (imageshape[1] - filtershape[1] + 1) / (pool_size[1] ) )

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape= (batch_size, nkerns[0], imageshape[0], imageshape[1]),
        filter_shape= (nkerns[1], nkerns[0], filtershape[0], filtershape[1]),#TODO
        poolsize= (2,2)#TODO
    )

    # Combine Layer 0 output and Layer 1 output
    # TODO: downsample the first layer output to match the size of the second
    # layer output.
    layer0_output_ds = downsample.max_pool_2d(
            input=layer0.output,
            ds= (2,2), # TDOD: change ds
            ignore_border = True
    )
    # concatenate layer
    layer2_input = T.concatenate([layer1.output, layer0_output_ds], axis=1)
                     
                 

    # TODO: Construct the third convolutional pooling layer
    imageshape[0] = ( (imageshape[0] - filtershape[0] + 1) / (pool_size[0] ) )
    imageshape[1] =  ( (imageshape[1] - filtershape[1] + 1) / (pool_size[1] ) )
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer2_input,
        image_shape= (batch_size, nkerns[0] + nkerns[1], imageshape[0], imageshape[1]),
        filter_shape= ( nkerns[2], nkerns[0] + nkerns[1], filtershape[0], filtershape[1]),
        poolsize= (2,2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[2] * 1 * 1).
    layer3_input = layer2.output.flatten(2)
    imageshape[0] = ( (imageshape[0] - filtershape[0] + 1) / (pool_size[0] ) )
    imageshape[1] =  ( (imageshape[1] - filtershape[1] + 1) / (pool_size[1] ) )
    
    rng = numpy.random.RandomState(1234)

    # construct a fully-connected sigmoidal layer
    
    layer3_1 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * imageshape[0] * imageshape[1],
        n_out= n_hidden,#TODO,
        activation=T.tanh
    )
    layer3_2 = HiddenLayer(
        rng,
        input=layer3_1.output,
        n_in=n_hidden,
        n_out= n_hidden,#TODO,
        activation=T.tanh
    )



    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3_2.output,
        n_in= n_hidden,#TODO
        n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # TODO: create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3_1.params + layer3_2.params + layer2.params + layer1.params + layer0.params#TODO

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)


if __name__ == "__main__":
    # test_lenet(verbose=True)
    # test_convnet(verbose=True)
    test_CDNN()
