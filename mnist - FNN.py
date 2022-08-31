import numpy as np
from keras.datasets import mnist
from math import sqrt

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class NeuralNetwork:
	# initialize the network-
	def __init__(self, layer_sizes, epoch, lr):
		self.weights = []
		self.biases = []
		self.z = [] # weights * inputs + bias for units in each layer
		self.a = [] # activation function applied to z for units in each layer
		self.alpha = lr
		self.epoch = epoch
		l = len(layer_sizes)
		for i in range(l-1):
			std = 1/sqrt((layer_sizes[i]+1)*layer_sizes[i+1])
			weights = np.random.normal(0,std,(layer_sizes[i+1], layer_sizes[i] +1))
			self.weights.append(weights[:,:-1])
			# We want a bias for each unit except the input layer!
			self.biases.append(weights[:,-1])



	def feedforward(self, input, label):
		output = 0
		self.a.append(input)
		for w,b in zip(self.weights, self.biases):
			z = np.dot(w, input) + b
			self.z.append(z)
			input = sigmoid(z)
			self.a.append(input)
		output = input
		self.error = np.power(output - label,2).sum(axis = -1)
		self.label = label
		self.output = output
		return output

	def dtanh(self, x):
		return 1.0 - np.tanh(x) ** 2

	def show_weights(self):
		print(self.weights)

	def backpropagate(self):
		delta = 2*(self.output-self.label) * sigmoid_prime(self.z[-1])
		delta0 = delta@self.weights[-1]

		self.weights[0] = self.weights[0] - self.alpha * np.outer(delta0, self.a[-3])
		self.biases[0] = self.biases[0] - self.alpha * delta0

		self.weights[-1] = self.weights[-1] - self.alpha * np.outer(delta, self.a[-2])
		self.biases[-1] = self.biases[-1] - self.alpha * delta

	def fit(self, X, y):
		#Trains a neural network on X, y
		n = len(X)
		for i in range(self.epoch):
			average_error = 0
			for _ ,(sample, label) in enumerate(zip(X,y)):
				sample = sample.flatten()
				label = np.eye(10)[label]
				self.feedforward(sample,label)
				average_error += self.error
				self.backpropagate()
				self.a.clear()
				self.z.clear()
			print("epoch #{} Error: {}" .format(i, average_error/n))

	def predict(self, X):
		#Computes the output of the trained network on the examples in X
		pred = 0
		for w, b in zip(self.weights, self.biases):
			z = np.dot(X, w.T) + b
			X = sigmoid(z)
			pred = X
		return np.argmax(pred)

	def score(self, X, y):
		"""Computes the average number of examples in X that the trained network classifies
		   incorrectly."""

		size = len(X)
		res = 0
		for sample, label in zip(X, y):
			sample = sample.flatten()
			if self.predict(sample) != label:
				res+=1
		return res / size

def mnist():
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    return (train_X, train_y), (test_X, test_y)

if __name__ == '__main__':
	np.random.seed(0)

	(train_X, train_y), (test_X, test_y) = mnist()

	#normalize the data :
	data_mean = np.mean(train_X,axis=0,keepdims=True)
	data_std = np.std(train_X,axis=0,keepdims=True)
	data_std[data_std==0]=1

	train_X = train_X.astype(np.float64) - data_mean
	train_X /= data_std

	test_X = test_X.astype(np.float64) - data_mean
	test_X /= data_std

	"""create neural network and train a model on mnist dataset:
	input layer(28*28=784) --> hidden layer (30) --> output layer(10)
	best number of epoch : 8
	best learning rate - 0.1
	**You can see all the attempts on "history_attempts.txt" file
	Best loss : 0.0942 %
	"""
	num_of_epochs = 8
	lr = 0.1
	nn = NeuralNetwork([784, 30, 10], num_of_epochs, lr)
	nn.fit(train_X,train_y)

	#calc the score of the model: print the average loss:
	loss = nn.score(test_X,test_y)
	print("loss: ",loss)

	f = open("history_attempts.txt", "a")
	f.write(f"Architecture: [784, 30, 10], number of epochs: {num_of_epochs}, learning rate : {lr}, loss : {loss}% \n")
	f.close()
