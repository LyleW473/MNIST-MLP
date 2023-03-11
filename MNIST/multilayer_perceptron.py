import numpy as np
from os.path import exists as os_path_exists
from os import mkdir as os_mkdir
from keras.datasets import mnist

"""
1. Load the dataset: Load the MNIST dataset using any available library.
2. Preprocess the data: Preprocess the dataset by normalizing the pixel values to between 0 and 1.
3. Initialize the weights and biases: Initialize the weights and biases of the neural network with random values.
4. Define the activation function: Choose an activation function to use in the hidden layers. A common choice is the ReLU activation function.
5. Define the loss function: Choose a loss function to use during training. For classification tasks like the MNIST dataset, cross-entropy loss is often used.
6. Define the forward pass: Define a function that takes the input data and propagates it through the neural network to produce a prediction.
7. Define the backward pass: Define a function that calculates the gradient of the loss function with respect to the weights and biases of the neural network.
8. Train the model: Train the neural network by repeatedly performing forward and backward passes on the training data and adjusting the weights and biases using an optimization algorithm like stochastic gradient descent.
9. Evaluate the model: Evaluate the performance of the trained neural network on the test set."""

"""Notes:
- "@" operator is used for conventional matrix multiplication
- Epoch = A single pass through the entire training dataset (i.e. All 60000 images)

"""

class MultilayerPerceptron:
    
    def __init__(self, learning_rate):

        # Load data
        self.load_mnist_data()
    
        self.number_of_correct_predictions = 0

        self.learning_rate = learning_rate

        # Initialise weights and biases
        self.initialise_w_and_b()

        # Train the network
        self.train(number_of_epochs = 6)

    def load_mnist_data(self):
        """
        x_train = The images of each number
        y_train = The labels for each image (to display what the image actually represents)
        """

        if (exists := os_path_exists("MNIST/data/x_train.npy")) == False:
            print("Loading data from keras")

            # Create data directory
            os_mkdir("MNIST/data")

            # Separate into training / testing sets
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
            
            # Normalise data so that pixel values are between 0 and 1
            self.x_train = self.x_train.reshape(60000, 784) / 255
            self.x_test = self.x_test.reshape(10000, 784) / 255

            # Convert labels into one-hot vectors
            """This is so that the label for each image is converted into a vector that can be compared to the final output of each epoch, to calculate the error/ cost"""
            self.y_train = np.eye(10)[self.y_train]
            self.y_test = np.eye(10)[self.y_test]

            # Save data
            np.save("MNIST/data/x_train.npy", self.x_train)
            np.save("MNIST/data/y_train.npy", self.y_train)
            np.save("MNIST/data/x_test.npy", self.x_test)
            np.save("MNIST/data/y_test.npy", self.y_test)

        elif exists == True:
            print("Loading from disk")

            # Loading from disk
            self.x_train = np.load("MNIST/data/x_train.npy")
            self.y_train = np.load("MNIST/data/y_train.npy")
            self.x_test = np.load("MNIST/data/x_test.npy")
            self.y_test = np.load("MNIST/data/y_test.npy")

    def calculate_correct_predictions(self, output, label):

        """Find the neuron with the largest value in the output and the label and see if they match (i.e. If the image was a 3 and the 3rd output neuron was the brightest, then this would be True)
        - int() converts False into 0, True into 1
        """
        self.number_of_correct_predictions += int(np.argmax(output) == np.argmax(label)) 
    
    def initialise_w_and_b(self):

        """ Shape should be:
        Number of neurons of the next layer, Number of neurons in the current layer
        (Right to left) [Results in faster computations]
        
        - Output layer should always have 10 neurons, as there can only be 10 values (values 0 - 9)
        """

        # Weights
        self.w_i_h1 = np.random.uniform(-0.5, 0.5, (100, 784))
        self.w_h1_o = np.random.uniform(-0.5, 0.5, (10, 100))
    
        # Biases
        self.b_i_h1 = np.zeros((100, 1))
        self.b_h1_o = np.zeros((10, 1))

    def test(self):
        
        for image, label in zip(self.x_test, self.y_test):
            """
            Convert images from vector "784" to a matrix (784, 1)
            Convert labels from vector "10" to a matrix (10, 1)
            """
            image.shape += (1,)
            label.shape += (1,)

            self.hidden_layer = self.forward_propagate(
                                                        biases = self.b_i_h1,
                                                        weights = self.w_i_h1,
                                                        input_values = image
                                                        )
            self.output_layer = self.forward_propagate(
                                                        biases = self.b_h1_o,
                                                        weights = self.w_h1_o,
                                                        input_values = self.hidden_layer
                                                        )
            
            self.calculate_correct_predictions(
                                                output = self.output_layer,
                                                label = label
                                                )
            
        print(f"Testing accuracy: {round((self.number_of_correct_predictions / self.x_test.shape[0]) * 100, 2)}%")

    def train(self, number_of_epochs):

        for _ in range(0, number_of_epochs):
            for image, label in zip(self.x_train, self.y_train):
                """
                Convert images from vector "784" to a matrix (784, 1)
                Convert labels from vector "10" to a matrix (10, 1)
                """
                image.shape += (1,)
                label.shape += (1,)

                self.hidden_layer = self.forward_propagate(
                                                            biases = self.b_i_h1,
                                                            weights = self.w_i_h1,
                                                            input_values = image
                                                            )
                self.output_layer = self.forward_propagate(
                                                            biases = self.b_h1_o,
                                                            weights = self.w_h1_o,
                                                            input_values = self.hidden_layer
                                                            )
                
                self.calculate_correct_predictions(
                                                    output = self.output_layer,
                                                    label = label
                                                    )

                # Update the weights and biases for all layers
                self.back_propagate(
                                    output = self.output_layer,
                                    label = label,
                                    image = image
                                    )
                
            print(f"Training accuracy: {round((self.number_of_correct_predictions / self.x_train.shape[0]) * 100, 2)}%")
            
            # Reset number of correct predictions for each epoch
            self.number_of_correct_predictions = 0

        # Test the network on the testing data
        self.test()
            
    """Activation function used to normalise values between 0 and 1"""
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    """Function used in backpropagation to compute the gradient of the loss function with respect to the weights and biases from the previous layer to the current layer
    - Tells you the slope / rate of change of the activation function at a particular input
    """
    def derivative_sigmoid(self, h):
        return h * (1 - h)
    
    def forward_propagate(self, biases, weights, input_values):

        """Forward propagates, finds the values for the neurons of the next layer"""

        # Matrix multiply the weights and the input values / neurons of the previous layer and add the bias
        x = biases + (weights @ input_values)

        # Sigmoid function to normalise values between 0 and 1
        next_layer_values = self.sigmoid(x = x)

        return next_layer_values
        
    def back_propagate(self, output, label, image):

        """Propagates backwards from the output layer to the input layer"""

        # Error / cost between the output and the actual label (what the image represents) (Uses prediction error)
        delta_o = (output - label)

        # Output layer to the hidden layer
        self.w_h1_o += - (self.learning_rate * delta_o @ np.transpose(self.hidden_layer))
        self.b_h1_o += - (self.learning_rate * delta_o)

        # Hidden layer to the input layer   
        
        # Error / cost between the hidden layer and the output layer
        delta_h = np.transpose(self.w_h1_o) @ delta_o * self.derivative_sigmoid(self.hidden_layer)

        """ 
        The gradient of the cost function indicates the direction of the steepest increase in the cost function, so with a "-" will flip the gradient
        to indicate the direction of the steepest decrease in the cost function.
        """
        self.w_i_h1 += - (self.learning_rate * delta_h @ np.transpose(image))
        self.b_i_h1 += - (self.learning_rate * delta_h)

multilayer_perceptron = MultilayerPerceptron(learning_rate = 0.02)