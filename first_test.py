import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 50, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 50, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    # network = regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

print "Let's Start"

env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500

model = neural_network_model(4)

for episode in range(2):
	env.reset()
	# this is each frame, up to 200...but we wont make it that far.
	for t in range(200):
		env.render()
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		print '---------------------'
		print observation
		print reward
		print done
		print info
		print '---------------------'
		
		print model.predict(observation.reshape(-1, 4, 1))
		
		if done:
			break
