# Power & Greed in Artificial Life
 
## Overview
An artistic exploration of how the complementary forces of trust and exploitation play out in a continuous Darwinian evolution. In this simulation, entities are initialized along a 2D grid with a neural network. Their color is determined in RGB by three factors:
+ R: Sexual Fitness, a value randomly assigned at birth and normally distributed among the population. Analogical to physical attractiveness.
+ G: Economic Fitness, The total of an individual's resources
+ B: Trustworthiness, A metric representing an individuals tendency to cooperate in two-player games with other individuals.

Depending on the composition of their neural network, each individual may have the ability to detect the color of their neighbors -- that is, they will be able to see how physically attractive, economically fit, and trustworthy they are. Based on this information, along with additional information about their position on the grid, their age, and other knowledge, each individual will decide on their action in the next time step. They may move, perhaps closer to a neighbor or resource. If they happen to cohabit a position on the grid with another neighbor, the neighbors may choose to reproduce and spawn a new entity derived from their genetic makeup at a random location on the grid, or engage in a two player game to earn resources. Their performance in this game will influence their color going forward. If they win the game by 'defecting' to the detriment of their neighbor, their resources will increase, but their trustworthiness will decrease. If they choose to cooperate, their trustworthiness will increase.

The goal is to find a meaningful set of parameters and create a simulation that will reach dynamically stable behavior over a long period of time. Ideally, certain 'phenotypes' would emerge the consequences of different 'survival strategies'.

A sample of the simulation is available in the MP4 file

## Code Walkthrough
The code begins with main. In main, the parameters for the simulation & individual entity are set. Then, the simulation is created. 

The main simulation involves initializing a dictionary of entities using the entity class. Each entity is initialized with a neural network and a set of inputs. In the think() function, it utilizes the neural network & inputs to compute outputs which drive its behavior in the next timestep. 

The main simulation also spawns resources along the 2D grid, which are represented as white squares. These resources are initialized in the 'Resource' class. Entities can collect resources by 'running over' them. You will note that each entities inputs contain the location of its nearest resource. 

The Game & Reproduction classes in the entity.py file show the mechanics of each. A game/ reproduction occurs only if both neighbors agree (ie. neither give a '0' in either the 'coop' or 'reproduce' output). 
This simulation took heavy inspiration from David Randal Miller's Bio Simulation, which I found on YouTube a while ago. 

