import gym
import random
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

class genetic_model(object):
	input_size = 0
	output_size = 0
	first_fully_connected_size = 0
	second_fully_connected_size = 0
	activation = 0

	def __init__(self, input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation):
		super(genetic_model, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.first_fully_connected_size = first_fully_connected_size
		self.second_fully_connected_size = second_fully_connected_size
		self.activation = activation

		self.input_layer = input_data(shape=[None, input_size, 1], name='input')
		self.first_fully_connected = fully_connected(self.input_layer, first_fully_connected_size, activation=activation)
		self.second_fully_connected = fully_connected(self.first_fully_connected, second_fully_connected_size, activation=activation)
		self.output = fully_connected(self.second_fully_connected, output_size, activation='softmax')
		self.network = regression(self.output, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
		self.model = tflearn.DNN(self.network, tensorboard_dir='log')
		
	def initialize_random_weights_and_biases(self):
		self.model.set_weights(self.first_fully_connected.W, np.random.rand(self.input_size, self.first_fully_connected_size))
		self.model.set_weights(self.first_fully_connected.b, np.random.rand(self.first_fully_connected_size))
		self.model.set_weights(self.second_fully_connected.W, np.random.rand(self.first_fully_connected_size, self.second_fully_connected_size))
		self.model.set_weights(self.second_fully_connected.b, np.random.rand(self.second_fully_connected_size))
		self.model.set_weights(self.output.W, np.random.rand(self.second_fully_connected_size, self.output_size))
		self.model.set_weights(self.output.b, np.random.rand(self.output_size))

def get_empty_generation(individuals_per_generation, input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation):
	current_generation_individuals = []
	for individual_index in xrange(0, individuals_per_generation):
		with tf.Graph().as_default():
			individual = genetic_model(input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation)
			individual.initialize_random_weights_and_biases()
			current_generation_individuals.append(individual)
	return current_generation_individuals

def train_new_generation(individuals_per_generation, input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation, X, Y, n_epoch, snapshot_step):
	print 'Training new generation....'
	current_generation_individuals = []
	for individual_index in xrange(0, individuals_per_generation):
		with tf.Graph().as_default():
			individual = genetic_model(input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation)
			individual.initialize_random_weights_and_biases()
			individual.model.fit({'input': X}, {'targets': Y}, n_epoch = n_epoch, snapshot_step = snapshot_step, show_metric = False)
			current_generation_individuals.append(individual)

	return current_generation_individuals

print '============'
print "Let's Start"
print '============'

env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500

input_size = 4
output_size = 2
first_fully_connected_size = 50
second_fully_connected_size = 50
activation = 'relu'
individuals_per_generation = 10

n_epoch = 5
snapshot_step = 500

# Initial generation
current_generation_individuals = get_empty_generation(individuals_per_generation, input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation)

current_generation = 0
best_performance = 0
# while best_performance < goal_steps:
while current_generation < 2:
	print '--------------------------'
	print 'Starting Generation: ' + str(current_generation)
	print '--------------------------'

	generation_scores = []
	generation_game_memory = []
	for individual in current_generation_individuals:
		individual_game_memory = []
		individual_performance = 0
		env.reset()

		for t in range(goal_steps):
			# env.render()

			if t == 0:
				action = env.action_space.sample()
			else:
				action_output = individual.model.predict(observation.reshape(-1, 4, 1))
				action = np.argmax(action_output[0])
				action_output = np.zeros(output_size)
				action_output[action] = 1
				individual_game_memory.append([observation, action_output])

			observation, reward, done, info = env.step(action)

			individual_performance = individual_performance + 1
			if done:
				break
		print 'Individual Performance: ' + str(individual_performance)
		generation_scores.append(individual_performance)
		generation_game_memory.append(individual_game_memory)

	best_performance = max(generation_scores)
	best_individual = generation_scores.index(best_performance)

	print '--------------------------'
	print 'Best score in this generation: ' + str(best_performance)
	print 'Best individual: ' + str(best_individual)

	current_generation = current_generation + 1

	X = [np.array([data[0]]).reshape(input_size, 1) for data in generation_game_memory[best_individual]]
	Y = [data[1] for data in generation_game_memory[best_individual]]

	current_generation_individuals = train_new_generation(individuals_per_generation, input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation, X, Y, n_epoch, snapshot_step)