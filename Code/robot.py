####################################
#      YOU MAY EDIT THIS FILE      #
# ALL OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from matplotlib import pyplot as plt
from scipy.stats import circstd

# Imports from this project
# You should not import any other modules, including config.py
# If you want to create some configuration parameters for your algorithm, keep them within this robot.py file
import config
import constants
from graphics import VisualisationLine

# Configure matplotlib for interactive mode
plt.ion()

# CONFIGURATION PARAMETERS. Add whatever configuration parameters you like here.
# Remember, you will only be submitting this robot.py file, no other files.
STARTING_MONEY=0
MONEY_DEMO_CUTOFF=13

MINIBATCH_SIZE=16
NUMBER_OF_NN_MODELS=4
NUM_EPOCHS=40
NUM_RETRAIN_EPOCHS=20

INITIAL_DEMO_LEN = 50
RESET_DEMO_LEN = 14
MAX_RECOVERY_DEMO = 20
MIN_RECOVERY_DEMO = 5

UNCERTAINTY_STD=0.4
MAX_ACTION_MAGNITUDE=2
MOVING_DISTANCE=0.2
MOVING_QUEUE_LEN=75

# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self):
        # The environment (only available during development mode)
        self.environment = None
        # A list of visualisations which will be displayed on the bottom half of the window
        self.visualisation_lines = []
        # Buffer for action model
        self.buffer_action = ReplayBuffer()
        # Action models
        self.action_models = [Action_Model(i+1) for i in range(NUMBER_OF_NN_MODELS)]
        # Random action to get out of uncertainty
        self.not_moving_before = False
        # Last 2 rewards
        self.prev_reward = 0
        self.prev_reward_changes = deque(maxlen=MOVING_QUEUE_LEN)
        # Minimum reward
        self.max_reward = -9999
        # Did we reset last time
        self.last_reset = False
        # money spent
        self.money_spent = {1:0,2:0,3:0}

    # Get the next training action
    def training_action(self, obs, money):
        global STARTING_MONEY
        #Set the Parameters
        if(STARTING_MONEY == 0): # Set global money parameter
            STARTING_MONEY = money
        recovery_demo_length = (round(money*3/4) - 10)*2
        demo_length = INITIAL_DEMO_LEN if money == STARTING_MONEY else max(min(MAX_RECOVERY_DEMO,recovery_demo_length), MIN_RECOVERY_DEMO)
        demo_length = RESET_DEMO_LEN if self.last_reset else demo_length
        
        # If completed, restart
        if (self.max_reward > -0.5):
            if (money < 5):
                print(f"Finished training. Money left: {money}. Money spent: {self.money_spent}")
                return 4, 0
            else:
                self.last_reset= True
                self.money_spent[2] += 5
                self.max_reward = -9999
                return 2, [0.05, np.random.rand()]

        # If requested demo_length is too expensive, ask for largest demo affordable
        if (money - 1 < 10+demo_length*0.5):
            demo_length = max(0, (round(money-1) - 10)*2)
            if(demo_length == 0): #Nothing more to learn
                print(f"Finished training. Money: {money}. Money spent: {self.money_spent}")
                return 4,0

        # Request demo at the beginning or at a reset
        if (STARTING_MONEY == money or self.last_reset):
            action_type = 3
            action_value = [0,demo_length]
            print(f"Requesting a demo of length {demo_length}. Money remaining: {money}")
            self.last_reset = False
            self.money_spent[3] += 10 + demo_length*0.5
            return action_type, action_value
        
        # Get n actions
        actions = np.array([model.predict_next_action(obs) for model in self.action_models])
        action_value, uncertain_action = self.average_action(actions)
        
        #if the average reward has not changed, request new action
        not_moving = False
        if(len(self.prev_reward_changes) == MOVING_QUEUE_LEN):
            not_moving = np.mean(self.prev_reward_changes) < MOVING_DISTANCE
            #Hopefully the robot will start moving, so delete movement history
            self.prev_reward_changes = deque(maxlen=MOVING_QUEUE_LEN)

        if (not_moving or uncertain_action) and demo_length > 0:
            # If not moving, give an opportunity to get out before resetting
            if(self.not_moving_before and not_moving):
                self.not_moving_before = False
                print(f"Resetting env")
                self.last_reset = True
                self.money_spent[2] += 5
                self.max_reward = -9999
                return 2, [0.05, np.random.rand()]
            
            action_type = 3
            action_value = [0,demo_length]
            print(f"Requesting a demo of length {demo_length}. Not_moving:{not_moving}. Not_moving_before:{self.not_moving_before} uncertain_action:{uncertain_action} Money remaining: {money}")
            self.not_moving_before = not_moving
            self.money_spent[3] += 10 + demo_length*0.5
            return action_type, action_value

        #print(f"Moving in direction {action_value}. Money remaining: {money}")
        if(self.environment):
            nextState = self.environment.dynamics(self.environment.state, action_value)
            self.visualisation_lines.append(VisualisationLine(self.environment.state[0], self.environment.state[1], nextState[0], nextState[1]))
        self.money_spent[1] += 0.002
        return 1, action_value

    def average_action(self, actions):
        angles = np.arctan2(actions[:,1], actions[:,0]) #Shape should be (4,)
        std = circstd(angles)
        
        if(self.environment):
            pass #print angles and straight lines

        return np.mean(actions, axis=0), std > UNCERTAINTY_STD

    # Get the next testing action
    def testing_action(self, obs):
        # Predict next observation and reward
        # Get n actions
        actions = np.array([model.predict_next_action(obs) for model in self.action_models])
        action, uncertain_action = self.average_action(actions) #Get best 3 actions
        return action

    # Receive a transition
    def receive_transition(self, obs, action, next_obs, reward):
        change = abs(self.prev_reward - reward)
        self.prev_reward_changes.append(change) # Automatically removes oldest reward
        self.prev_reward = reward

        self.max_reward = reward if reward > self.max_reward else self.max_reward

    # Receive a new demonstration
    def receive_demo(self, demo):
        
        if self.environment:
            state = self.environment.state

        for obs, action in demo:
            self.buffer_action.add_data(obs, action)

            if self.environment:
                nextState = self.environment.dynamics(state, action)
                self.visualisation_lines.append(VisualisationLine(state[0], state[1], nextState[0], nextState[1], colour="red"))
                state=nextState

        #Train 4 NN models
        for model in self.action_models:
            epochs = NUM_EPOCHS if len(demo) == INITIAL_DEMO_LEN else NUM_RETRAIN_EPOCHS
            model.train(self.buffer_action, epochs)


# ReplayBuffer class stores transitions
class ReplayBuffer:
    def __init__(self):
        self.features = []
        self.labels = []
        self.size = 0

    def add_data(self, state, action):
        self.features.append(state)
        self.labels.append(action)
        self.size += 1

    # Create minibatches for a single epoch of training (one epoch means all the training data is seen once)
    def sample_epoch_minibatches(self, minibatch_size):
        # Convert lists to NumPy arrays for indexing
        states_array = np.array(self.features, dtype=np.float32)
        actions_array = np.array(self.labels, dtype=np.float32)
        # Shuffle indices
        indices = np.random.permutation(self.size)
        minibatches = []
        # Create minibatches
        for i in range(0, self.size, minibatch_size):
            # Get the indices for this minibatch
            minibatch_indices = indices[i: i + minibatch_size]
            minibatch_states = states_array[minibatch_indices]
            minibatch_actions = actions_array[minibatch_indices]
            # Convert to torch tensors
            inputs = torch.from_numpy(minibatch_states)
            targets = torch.from_numpy(minibatch_actions)
            minibatches.append((inputs, targets))
        return minibatches

# Policy is used to predict the next action
class Action_Model:

    def __init__(self, model_number):
        self.network = Network(5, 64, 2)
        self.optimiser = optim.Adam(self.network.parameters(), lr=0.005)
        self.loss_fn = nn.MSELoss()
        self.losses = []
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Num Epochs')
        self.ax.set_ylabel('Loss')
        self.ax.set_title(f'Action Loss Curve; NN {model_number}')
        self.ax.set_yscale('log')
        self.line, = self.ax.plot([], [], linestyle='-', marker=None, color='blue')
        plt.show()

    def train(self, action_buffer, num_epochs=1):
        for epoch in range(num_epochs):
            loss_sum = 0
            minibatches = action_buffer.sample_epoch_minibatches(MINIBATCH_SIZE)
            for inputs, targets in minibatches:
                # Set the network to training mode
                self.network.train()
                # Forward pass: compute predicted next states
                predictions = self.network.forward(inputs)
                # Compute the loss
                loss = self.loss_fn(predictions, targets)
                # Backward pass and optimization step
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                # Get the loss value and add to the list of losses
                loss_value = loss.item()
                loss_sum += loss_value

            #plot
            ave_loss = loss_sum / len(minibatches)
            self.losses.append(ave_loss)
                # Plot the loss curve
            self.line.set_xdata(range(1, len(self.losses) + 1))
            self.line.set_ydata(self.losses)
            # Adjust the plot limits
            self.ax.relim()
            self.ax.autoscale_view()
            # Redraw the figure
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def predict_next_action(self, obs):
        input = torch.tensor(obs, dtype=torch.float32)
        # Set model to evaluation mode
        self.network.eval()
        with torch.no_grad():
            # Forward pass
            prediction_tensor = self.network(input)
        # Remove batch dimension and convert to numpy
        prediction = prediction_tensor.squeeze(0).numpy()
        return prediction

# This is the network that is trained on the transition data
class Network(nn.Module):

    # Initialise
    def __init__(self, input_size, hidden_size, output_size):
        super(Network, self).__init__()
        # Define the first hidden layer
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.ReLU()
        # Define the second hidden layer
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.activation2 = nn.ReLU()
        # Define the third hidden layer
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.activation3 = nn.ReLU()
        # Define the output layer
        self.output = nn.Linear(hidden_size, output_size)

    # Forward pass
    def forward(self, input):
        # Pass data through the layers
        x = self.hidden1(input)
        x = self.activation1(x)
        x = self.hidden2(x)
        x = self.activation2(x)
        #x = self.hidden3(x)
        #x = self.activation3(x)
        # Pass data through output layer
        output = self.output(x)
        return output

# # Policy is used to predict the next action
# class Dynamics_Model:

#     def __init__(self):
#         self.network = Network(5, 20, 6)
#         self.optimiser = optim.Adam(self.network.parameters(), lr=0.005)
#         self.loss_fn = nn.MSELoss()
#         self.losses = []
#         self.fig, self.ax = plt.subplots()
#         self.ax.set_xlabel('Num Epochs')
#         self.ax.set_ylabel('Loss')
#         self.ax.set_title('Dynamics Loss Curve')
#         self.ax.set_yscale('log')
#         self.line, = self.ax.plot([], [], linestyle='-', marker=None, color='blue')
#         plt.show()

#     def train(self, dynamics_buffer, num_epochs=1):
#         for epoch in range(num_epochs):
#             loss_sum = 0
#             minibatches = dynamics_buffer.sample_epoch_minibatches(MINIBATCH_SIZE)
#             for inputs, targets in minibatches:
#                 # Set the network to training mode
#                 self.network.train()
#                 # Forward pass: compute predicted next states
#                 predictions = self.network.forward(inputs)
#                 # Compute the loss
#                 loss = self.loss_fn(predictions, targets)
#                 # Backward pass and optimization step
#                 self.optimiser.zero_grad()
#                 loss.backward()
#                 self.optimiser.step()
#                 # Get the loss value and add to the list of losses
#                 loss_value = loss.item()
#                 loss_sum += loss_value

#             #plot
#             ave_loss = loss_sum / len(minibatches)
#             self.losses.append(ave_loss)
#                 # Plot the loss curve
#             self.line.set_xdata(range(1, len(self.losses) + 1))
#             self.line.set_ydata(self.losses)
#             # Adjust the plot limits
#             self.ax.relim()
#             self.ax.autoscale_view()
#             # Redraw the figure
#             self.fig.canvas.draw()
#             self.fig.canvas.flush_events()

#     def predict_next_obs_and_reward(self, obs):
#         input = torch.tensor(obs, dtype=torch.float32)
#         # Set model to evaluation mode
#         self.network.eval()
#         with torch.no_grad():
#             # Forward pass
#             prediction_tensor = self.network(input)
#         # Remove batch dimension and convert to numpy
#         prediction = prediction_tensor.squeeze(0).numpy()
#         return prediction