"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int, # dimensions of 
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        num_classes: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = num_classes
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
        self.m = {}
        self.v = {}
        for i in range(1, self.num_layers + 1):
            self.m["W" + str(i)] = np.zeros_like(self.params['W' + str(i)])
            self.m["b" + str(i)] = np.zeros_like(self.params['b' + str(i)])
            self.v["W" + str(i)] = np.zeros_like(self.params['W' + str(i)])
            self.v["b" + str(i)] = np.zeros_like(self.params['b' + str(i)])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        output = X @ W + b
        return output

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0,X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        return np.where(X <= 0, 0, 1)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:

      # TODO ensure that this is numerically stable
      sig = 1 / (1+np.exp(-x))
      return sig

    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
      return X * (1-X)

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
      # TODO implement this
      diff_array = np.subtract(y,p)
      squared_array = np.square(diff_array)
      mse_value  = squared_array.mean()
      return mse_value

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.
        #Inputs from X 
        self.outputs["X0"] = X # Linear Outputs
        self.outputs["A0"] = X # Activations
        let_X = X
        for layer in range(1, self.num_layers + 1):
            W: np.ndarray = self.params["W" + str(layer)]
            b: np.ndarray = self.params["b" + str(layer)]
            linear_output = self.linear(W, let_X, b)
            self.outputs["X" + str(layer)] = linear_output
            if layer == self.num_layers:
                Activations = self.sigmoid(linear_output)
            else:
                Activations = self.relu(linear_output)
            self.outputs["A" + str(layer)] = Activations
            let_X = Activations
        return self.outputs["A" + str(self.num_layers)] # Returns the prediction

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}

        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.sigmoid_grad if it helps organize your code.
        m = y.shape[0]
        loss = self.mse(y,self.outputs["A" + str(self.num_layers)])

        #grad = self.mse_derivative(y,self.outputs["A" + str(self.num_layers)])
  
        grad = (2/y.size)*(self.outputs["A" + str(self.num_layers)] - y)

        for i in range(self.num_layers,0,-1):
            if i == self.num_layers:
                grad = np.multiply(grad,self.sigmoid_grad(self.outputs["A"+str(self.num_layers)]))
                self.gradients["b"+str(i)] = np.sum(grad, axis=0)
                delta = np.dot(self.outputs["A" + str(i-1)].T,grad)
                grad = np.dot(grad, self.params['W' + str(i)].T)

            else:
                grad = np.multiply(grad,self.relu_grad(self.outputs["X" + str(i)]))
                delta = np.dot(self.outputs["A" + str(i-1)].T,grad) 
                self.gradients["b"+str(i)] = np.sum(grad, axis=0)
                grad = np.dot(grad, self.params['W' + str(i)].T)
            self.gradients["W"+str(i)] = delta 
        return loss
        
    def update(
        self,
        lr: float ,
        b1: float ,
        b2: float ,
        eps: float ,
        opt: str ,
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        self.t=0

        if opt == "ADAM":

            self.t += 1
            for i in range(1, self.num_layers + 1):
                #print("shape of M",self.m["m" + str(i)].shape )
                #print("shape of W",self.gradients['W' + str(i)].shape )
                self.m["W" + str(i)] = b1 * self.m["W" + str(i)] + (1.0 - b1) * self.gradients['W' + str(i)]
                self.m["b" + str(i)] = b1 * self.m["b" + str(i)] + (1.0 - b1) * self.gradients['W' + str(i)]
                self.v["W" + str(i)] = b2 * self.v["W" + str(i)] + (1.0 - b2) * (self.gradients['W' + str(i)] ** 2)
                self.v["b" + str(i)] = b2 * self.v["b" + str(i)] + (1.0 - b2) * (self.gradients['W' + str(i)] ** 2)
                # Compute bias-corrected moment estimates
                m_hat_W = self.m["W" + str(i)]  / (1 - b1**self.t)
                m_hat_b = self.m["b" + str(i)]  / (1 - b1**self.t)
                v_hat_W = self.v["W" + str(i)] / (1 - b2**self.t)
                v_hat_b = self.v["b" + str(i)] / (1 - b2**self.t)

                
                self.params['W' + str(i)] -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
                #self.params['b' + str(i)] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
                self.params['b' + str(i)] -= lr * self.gradients['b' + str(i)]
        else:
            for i in range(1, self.num_layers + 1):
                self.params['W' + str(i)] -= lr * self.gradients['W' + str(i)]
                self.params['b' + str(i)] -= lr * self.gradients['b' + str(i)]