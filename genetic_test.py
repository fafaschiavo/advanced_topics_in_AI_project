import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

class genetic_model(object):
	def __init__(self, input_size, first_fully_connected_size, second_fully_connected_size, activation):
		super(genetic_model, self).__init__()
		self.input_layer = input_data(shape=[None, input_size, 1], name='input')
		self.first_fully_connected = fully_connected(self.input_layer, first_fully_connected_size, activation=activation)
		self.second_fully_connected = fully_connected(self.first_fully_connected, second_fully_connected_size, activation=activation)
		self.network = fully_connected(self.second_fully_connected, 2, activation='softmax')
		self.model = tflearn.DNN(self.network, tensorboard_dir='log')
		

print "Let's Start"

env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500

genetic_model = genetic_model(4, 50, 50, 'relu')

# to set the weights
genetic_model.model.set_weights(genetic_model.first_fully_connected.W, np.random.rand(4, 50))

# # to print the weghts
# print genetic_model.model.get_weights(genetic_model.first_fully_connected.W)

print genetic_model.model.get_weights(genetic_model.first_fully_connected.b)
genetic_model.model.set_weights(genetic_model.first_fully_connected.b, np.random.rand(50))
print genetic_model.model.get_weights(genetic_model.first_fully_connected.b)
