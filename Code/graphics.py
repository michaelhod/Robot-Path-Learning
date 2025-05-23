#########################
# DO NOT EDIT THIS FILE #
#########################

# Imports from external libraries
import numpy as np
import pygame
from scipy.ndimage import zoom

# Imports from this project
import constants
import config


class VisualisationLine:
    def __init__(self, x1, y1, x2, y2, colour=(255, 255, 255), width=0.01):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.colour = colour
        self.width = width


# The Graphics class performs all the pygame drawing
class Graphics:

    # Initialisation of new graphics
    def __init__(self, environment):
        # Screen dimensions
        self.screen = pygame.display.set_mode((constants.ENVIRONMENT_WIDTH * config.WINDOW_SIZE + 2 * constants.WINDOW_MARGIN, 2 * constants.ENVIRONMENT_HEIGHT * config.WINDOW_SIZE + 3 * constants.WINDOW_MARGIN))
        # Create the environment's background image for the dynamics
        image_orig = np.dstack([environment.resistance * 255] * 3)
        old_width, old_height, channels = image_orig.shape
        scale_factor = config.WINDOW_SIZE / old_height
        zoom_factors = (scale_factor, scale_factor, 1)
        image_resized = zoom(image_orig, zoom_factors, order=1)
        image_resized = image_resized.astype(image_orig.dtype, copy=False)
        image_flipped = image_resized[:, ::-1, :]
        self.environment_image = image_flipped
        # Create the top and bottom canvas
        self.top_canvas = pygame.surfarray.make_surface(self.environment_image)
        self.bottom_canvas = pygame.surfarray.make_surface(self.environment_image)
        # Set a window title
        pygame.display.set_caption("Robot Learning")
        # Clock to control the frame rate
        self.clock = pygame.time.Clock()

    # Function to draw the environment, and any visualisations, on the window
    def draw(self, environment, visualisation_lines):
        # Clear the screen
        self.screen.fill((0, 0, 0))
        # Draw the top panel
        self.top_canvas = pygame.surfarray.make_surface(self.environment_image)
        self.draw_finish(self.top_canvas)
        self.draw_border(self.top_canvas)
        self.draw_robot(environment, self.top_canvas)
        self.screen.blit(self.top_canvas, (constants.WINDOW_MARGIN, constants.WINDOW_MARGIN))
        # Draw the bottom panel
        self.bottom_canvas = pygame.surfarray.make_surface(self.environment_image)
        self.draw_finish(self.bottom_canvas)
        self.draw_border(self.bottom_canvas)
        self.draw_robot(environment, self.bottom_canvas)
        self.draw_visualisation_lines(visualisation_lines, self.bottom_canvas)
        self.screen.blit(self.bottom_canvas, (constants.WINDOW_MARGIN, config.WINDOW_SIZE + 2 * constants.WINDOW_MARGIN))
        # Update the display
        pygame.display.flip()
        # Tick the clock, i.e. wait for one step of the environment
        self.clock.tick(constants.FRAME_RATE)

    # Function to draw a border around the environment
    def draw_border(self, canvas):
        # Draw a rectangle to show the border
        pygame.draw.rect(canvas, (250, 200, 200), pygame.Rect(0, 0, constants.ENVIRONMENT_WIDTH * config.WINDOW_SIZE, config.WINDOW_SIZE), 5)

    # Function to draw the robot
    def draw_robot(self, environment, canvas):
        # Draw the robot
        position = self.world_pos_to_window_pos(environment.state)
        radius = self.world_len_to_window_len(constants.ROBOT_RADIUS)
        pygame.draw.circle(canvas, constants.ROBOT_COLOUR, position, radius)

    # Draw the finishing line
    def draw_finish(self, canvas):
        start = self.world_pos_to_window_pos((constants.GOAL_LINE_X, 0))
        end = self.world_pos_to_window_pos((constants.GOAL_LINE_X, 1.0))
        pygame.draw.line(canvas, (200, 0, 0), start, end, 5)

    # Function to draw any visualisation lines
    def draw_visualisation_lines(self, visualisation_lines, canvas):
        for visualisation in visualisation_lines:
            # For each visualisation, get the attributes necessary to create a pygame line
            start_pos = self.world_pos_to_window_pos([visualisation.x1, visualisation.y1])
            end_pos = self.world_pos_to_window_pos([visualisation.x2, visualisation.y2])
            width = self.world_len_to_window_len(visualisation.width)
            pygame.draw.line(canvas, visualisation.colour, start_pos, end_pos, width)

    # Function to covert a position in world/environment space, to a position in pixels on the window
    def world_pos_to_window_pos(self, world_pos):
        window_pos_x = int(config.WINDOW_SIZE * world_pos[0])
        window_pos_y = int(config.WINDOW_SIZE - config.WINDOW_SIZE * world_pos[1])
        return window_pos_x, window_pos_y

    # Function to convert a length in world/environment space, to a length in pixels on the window
    def world_len_to_window_len(self, world_length):
        window_length = int(config.WINDOW_SIZE * world_length)
        return window_length
