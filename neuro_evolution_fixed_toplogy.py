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
		self.model = tflearn.DNN(self.output, tensorboard_dir='log')
		
	def initialize_random_weights_and_biases(self):
		self.model.set_weights(self.first_fully_connected.W, np.random.rand(self.input_size, self.first_fully_connected_size))
		self.model.set_weights(self.first_fully_connected.b, np.random.rand(self.first_fully_connected_size))
		self.model.set_weights(self.second_fully_connected.W, np.random.rand(self.first_fully_connected_size, self.second_fully_connected_size))
		self.model.set_weights(self.second_fully_connected.b, np.random.rand(self.second_fully_connected_size))
		self.model.set_weights(self.output.W, np.random.rand(self.second_fully_connected_size, self.output_size))
		self.model.set_weights(self.output.b, np.random.rand(self.output_size))

	def generate_mutations(self, mutation_rate):
		genes_to_mutate = [
			self.first_fully_connected.W,
			self.first_fully_connected.b,
			self.second_fully_connected.W,
			self.second_fully_connected.b,
			self.output.W,
			self.output.b,
		]

		for weight_portion in genes_to_mutate:
			gene = self.model.get_weights(weight_portion)

			if len(gene.shape) == 1:
				for i in range(0, gene.shape[0]):
					will_mutate = random.uniform(0, 1)
					if will_mutate < mutation_rate:
						new_weight = random.uniform(0, 1)
						gene[i] = new_weight
				self.model.set_weights(weight_portion, gene)

			if len(gene.shape) == 2:
				for i in range(0, gene.shape[0]):
					for j in range(0, gene.shape[1]):
						will_mutate = random.uniform(0, 1)
						if will_mutate < mutation_rate:
							new_weight = random.uniform(0, 1)
							gene[i][j] = new_weight
				self.model.set_weights(weight_portion, gene)

print "Let's Start"

env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500

input_size = 4
output_size = 2
first_fully_connected_size = 50
second_fully_connected_size = 50
activation = 'relu'
individuals_per_generation = 30
mutation_rate = 0.2
episodes_per_individual = 10

# Initial generation
current_generation_individuals = []
for individual_index in xrange(0, individuals_per_generation):
	with tf.Graph().as_default():
		individual = genetic_model(input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation)
		individual.initialize_random_weights_and_biases()
		current_generation_individuals.append(individual)
		# individual.generate_mutations(mutation_rate)

current_generation = 0
while True:
	print '--------------------------'
	print 'Current Generation: ' + str(current_generation)
	print '--------------------------'

	generation_scores = []
	for individual in current_generation_individuals:
		individual_performance = 0
		for episode in xrange(0,episodes_per_individual):
			env.reset()
			# this is each frame, up to 200...but we wont make it that far.
			for t in range(goal_steps):
				# env.render()

				if t == 0:
					action = env.action_space.sample()
				else:
					action_output = individual.model.predict(observation.reshape(-1, 4, 1))
					action = np.argmax(action_output[0])
					# print action_output

				observation, reward, done, info = env.step(action)

				action_output = individual.model.predict(observation.reshape(-1, 4, 1))

				individual_performance = individual_performance + 1
				if done:
					break
		print 'Individual Performance: ' + str(individual_performance)
		generation_scores.append(individual_performance)