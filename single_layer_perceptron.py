import numpy as np, random

"""Notes:
- Single layer perceptron with 1 neuron, 3 inputs and 1 output
- The correct answer should always be the first digit in the permutation

- Permutations like (0, 0, 0) and (1, 1, 1) do not work because all inputs will be the exact same. This would result in all weights being adjusted by the same amount
- (This is bceause if all inputs have the same value, the dot product between the input vector and the weight vector will always be the same)
"""

# Sigmoid function to normalise, so the result is between 0 and 1
"""Equal to:  1 / (1 + e^-x)"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function to find the gradient
"""
- This is because on the sigmoid curve, larger positive / negative numbers have a shallow gradient
- Larger numbers will signify that the neuron was confident that it was correct
- Multiplying by the gradient will mean that the adjustment is greater at "less confident" numbers
 """
def derivative_sigmoid(x):
    return x * (1 - x)

# Calculates the weighted sum of the neuron's inputs and finds the output using the sigmoid function
"""Equal to:  1 / (1 + e^-x), where x is the weighted sum of the neuron's inputs"""
def think(inputs, weights):
    return sigmoid(x = np.dot(inputs, weights))

# Trains the neural network
def train(inputs, actual_outputs, num_of_iterations, weights, learning_rate):
    
    print("Training commenced")

    for i in range(0, num_of_iterations):
        
        # Find the answer that the neural network thinks is correct
        output = think(inputs = inputs, weights = weights)

        # Calculate the error for each example
        error = actual_outputs - output

        # Multiply the error by the input and the gradient of the Sigmoid curve (derivative_sigmoid) to find the adjustment
        """ Widrow Hoff learning algorithm
        - Less confident weights are adjusted more
        - Inputs that are zero cause no changes to weights
        """
        adjustment = np.dot(inputs.T, learning_rate * error * derivative_sigmoid(output))

        # Adjust weights
        weights += adjustment
        
        if i % (num_of_iterations / 5) == 0:
            print(f"Output: {output}")
            print(f"Error: {error}")
            print()

# Seed to ensure random numbers are constant
random.seed(1)

# 3 inputs (1 input for each digit), 1 output (The output is always the leftmost input (i.e. the first digit))
weights = np.random.uniform(-1, 1, size = (3, 1))

training_permutations = [
                    (0, 0, 1),
                    (0, 1, 0),
                    (1, 0, 0), 
                    (1, 0, 1)
                    ]

testing_permutations = [
                        (0, 1, 1),
                        (1, 1, 0)
                        ]

training_set_inputs = np.array(training_permutations)
training_set_outputs = np.array([[training_permutation[0] for training_permutation in training_permutations]]).T

print(f"Correct outputs: {training_set_outputs}\n")
print(f"Starting weights: {weights}")

train(
    inputs = training_set_inputs, 
    actual_outputs = training_set_outputs, 
    num_of_iterations = 50000,
    weights = weights,
    learning_rate = 0.5
    )
# Testing:
testing_set_inputs = np.array(testing_permutations)
testing_set_outputs = np.array([[testing_permutation[0] for testing_permutation in testing_permutations]]).T

print(f"New situation tests: {testing_set_inputs}\n")

output = think(inputs = testing_set_inputs, weights = weights)
print(f"Testing output: {output}")
print(f"Expected output: {testing_set_outputs}")