####################################
#      YOU MAY EDIT THIS FILE      #
# ALL OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

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
STARTING_MONEY=100
STARTING_MONEY_SET = False
MINIBATCH_SIZE=16


# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self):
        # The environment (only available during development mode)
        self.environment = None
        # A list of visualisations which will be displayed on the bottom half of the window
        self.visualisation_lines = []
        # Most recent demo provided
        self.current_demo = []
        # Current step of demo
        self.current_step = 0
        # Number of epochs run to train action model
        self.action_epoch_count = 0
        # Buffer for dynamics model
        self.buffer_dynamics = ReplayBuffer()
        # Buffer for action model
        self.buffer_action = ReplayBuffer()
        # Dynamics model
        self.dynamics_model = Dynamics_Model()
        # Action model
        self.action_model = Action_Model()
        # Has action buffer been prepped
        self.has_prep_action_buffer = False
        # Last reward recieved
        self.best_reward = -99999999

    # Get the next training action
    def training_action(self, obs, money):
        global STARTING_MONEY, STARTING_MONEY_SET

        #Set the Parameters
        if(not STARTING_MONEY_SET): # Set global money parameter
            STARTING_MONEY = money
            STARTING_MONEY_SET = True
        money_demo_cutoff = STARTING_MONEY // 20 # 5
        demo_length = int (min(20, (40 * money) // STARTING_MONEY)) #Between 20 and 8 
        train_dynamics_model_every_n_actions = 1
        dynamics_epochs = 1
        
        # Train the model every 20 steps
        if(self.buffer_dynamics.size > 16 and self.buffer_dynamics.size % train_dynamics_model_every_n_actions == 0):
            self.dynamics_model.train(self.buffer_dynamics, dynamics_epochs)

        # If no money left or the last reward was close to endpoint, finish training
        if(money < 39.5 or self.action_epoch_count > 100):
            return 4, 0
        
        # If not much money, finish demo execution and then train an action model epoch
        elif(money < money_demo_cutoff or self.best_reward > -0.1):
            if(len(self.current_demo) > self.current_step):
                action_type = 1
                action_value = self.current_demo[self.current_step][1]
                self.current_step += 1
            else:
                if(not self.has_prep_action_buffer):
                    self.prep_action_buffer()
                    self.has_prep_action_buffer = True
                self.action_model.train(self.buffer_action) #Run one epoch of training each time to check money cost
                self.action_epoch_count += 1
                return 0, 0
        
        # If no demo actions left, choose "Request demo"
        elif(len(self.current_demo) == self.current_step):
            action_type = 3
            action_value = [0,demo_length]
            print(demo_length)

        # If abundant money and still need to run a demo, do action
        else:
            action_type = 1
            action_value = self.current_demo[self.current_step][1]
            self.current_step += 1

        return action_type, action_value

    # Get the next testing action
    def testing_action(self, obs):
        # Predict next observation and reward
        nextobs_reward = self.dynamics_model.predict_next_obs_and_reward(obs)
        obs_nextobs_reward = np.append(obs, nextobs_reward)
        action = self.action_model.predict_next_action(obs_nextobs_reward)
        return action

    # Receive a transition
    def receive_transition(self, obs, action, next_obs, reward):
        self.best_reward = reward if reward > self.best_reward else self.best_reward
        
        inputData = obs
        outputData = np.append(next_obs, reward)
        self.buffer_dynamics.add_data(inputData, outputData)

    # Receive a new demonstration
    def receive_demo(self, demo):
        for obs, action in demo:
            self.buffer_action.add_data(obs, action)
        self.current_demo = demo
        self.current_step = 0

    def prep_action_buffer(self):
        temp = ReplayBuffer()
        
        for i in range(self.buffer_action.size):
            obs = self.buffer_action.features[i]
            labels = self.buffer_action.labels[i]

            nextobs_reward = self.dynamics_model.predict_next_obs_and_reward(obs)
            feature = np.append(obs, nextobs_reward)

            temp.add_data(feature, labels)

        self.buffer_action = temp


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

    def __init__(self):
        self.network = Network(11, 50, 2)
        self.optimiser = optim.Adam(self.network.parameters(), lr=0.005)
        self.loss_fn = nn.MSELoss()
        self.losses = []
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Num Epochs')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Action Loss Curve')
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

    def predict_next_action(self, obs_nextobs_reward):
        input = torch.tensor(obs_nextobs_reward, dtype=torch.float32)
        # Set model to evaluation mode
        self.network.eval()
        with torch.no_grad():
            # Forward pass
            prediction_tensor = self.network(input)
        # Remove batch dimension and convert to numpy
        prediction = prediction_tensor.squeeze(0).numpy()
        return prediction

# Policy is used to predict the next action
class Dynamics_Model:

    def __init__(self):
        self.network = Network(5, 20, 6)
        self.optimiser = optim.Adam(self.network.parameters(), lr=0.005)
        self.loss_fn = nn.MSELoss()
        self.losses = []
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Num Epochs')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Dynamics Loss Curve')
        self.ax.set_yscale('log')
        self.line, = self.ax.plot([], [], linestyle='-', marker=None, color='blue')
        plt.show()

    def train(self, dynamics_buffer, num_epochs=1):
        for epoch in range(num_epochs):
            loss_sum = 0
            minibatches = dynamics_buffer.sample_epoch_minibatches(MINIBATCH_SIZE)
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

    def predict_next_obs_and_reward(self, obs):
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
        x = self.hidden3(x)
        x = self.activation3(x)
        # Pass data through output layer
        output = self.output(x)
        return output