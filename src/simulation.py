from matplotlib.animation import FuncAnimation
import numpy as np
import scipy as sp
import os
import random
import matplotlib.pyplot as plt
import copy

from entity import Entities, Reproduction, Game
from resource import Resource

# Simulation class
class Simulation:
    def __init__(self, sim_params, entity_params, save_animation=False, animation_name = None, play_GIF=False):

        # Boolean for if we save the animation
        self.save_animation = save_animation

        # Name to save the file
        self.animation_name = animation_name

        # play Gif or not?
        self.play_GIF = play_GIF

        # contains the ordered list of grids
        self.grids = []

        # Initialize entities
        self.entities = Entities(sim_params, entity_params)
        
        # Other attributes of interest
        self.num_entities = sim_params['initial_entities']
        self.grid_size = sim_params['x_dim'] * sim_params['y_dim']
        
        # Initialize dictionary to store locations of the resources
        res_locations = {} 
        
        # will allow fast location generation in high population density
        free_squares = set()
        
        '''
        MAIN LOOP:
        '''
        # Initialize Grid
        for i in range(sim_params['iterations']):
            if (len(self.entities.ent_dict) == 0):
                break
            background_color = np.array([245, 245, 245], dtype=np.uint8)
            self.grid = np.full(((sim_params['y_dim']), (sim_params['x_dim']), 3), background_color, dtype=np.uint8)
            for i in range(sim_params['x_dim']):
                for j in range(sim_params['y_dim']):
                    free_squares.add((i, j))

            # dead_ents: set of entities whose health <= 0 after simulation, or have been killed off. Initiated outside of loop. key: coordinate, value:id ex {(x_loc, y_loc) : id_code}
            dead_ents = {}
            
            # new_ents: dictionary of new entities spawned from reproduction
            new_ents = {}

            # ent_locations: key: entity location , value: set of entities at that location {(x_loc, y_loc): {id_1, id_2, id_3...}}
            ent_locations = {}
            
            # collision_locations: key:location, value:list of entity id_codes at that location {(x_loc, y_loc): [id_1, id_2, id_3...]}
            collision_locations = set()
                       
            # Unpack current entities dictionary and add locations to locations dictionaries
            for id_code, entity in self.entities.ent_dict.items():
                ent_location = (entity.inputs['x_loc'], entity.inputs['y_loc'])
                # Check if entity picks up a resource:
                if ent_location in res_locations:
                    entity.inputs['G'] += res_locations[ent_location].value
                    del res_locations[ent_location]
                if ent_location not in ent_locations:
                    ent_locations[ent_location] = set()
                ent_locations[ent_location].add(id_code)
                if ent_location in free_squares:
                    free_squares.remove(ent_location)
                
                # If there is already an entity at that location, store the location in a set
                if len(ent_locations[ent_location]) > 1:
                    if ent_location not in collision_locations:
                        collision_locations.add(ent_location)

            # collision_events: key: id_code of entity in pairing; value: its pair; {id_code: id_code_2}
            # Purpose of this code/dictionary is to create/store pairings of interacting entities
            collision_events = {}
            # For each collision location, pair up the first two entities and kill off the rest
            for location in collision_locations:
                entities = list(ent_locations[location])
                collision_events[entities[0]] = entities[1] # Allow the first two entities to collide
                # Kill off the rest of the entities at that location
                if location not in dead_ents:
                    dead_ents[location] = []
                dead_ents[location].extend(entities[2:])

            # Spawn new resources if there's enough room on the grid
            if len(free_squares) > 0:
                
                # res_spawn_rate variable controls how much food will spawn
                for i in range(sim_params['res_spawn_rate']):
                    
                    # Initialize random location for resource
                    res_loc = (np.random.randint(0, sim_params['x_dim']), np.random.randint(0, sim_params['y_dim']))
                    
                    # Generate random locations until one is found that is unique
                    res_loc = random.sample(tuple(free_squares), 1)[0]
                    free_squares.remove(res_loc)
                    res_locations[res_loc] = Resource(8, res_loc[0], res_loc[1])
                    
            # Temporary line to display food in grid -- food will be white (255, 255, 255)
            for key, value in res_locations.items():
                self.grid[key[1],key[0]]=[np.uint8(255.0),np.uint8(255.0),np.uint8(255.0)]
        
            # Initialize KD Tree between entities
            ent_locations_list = list(ent_locations.keys())
            kd_tree_entities = sp.spatial.KDTree(ent_locations_list) 
            
            # Initialize KD Tree for nearest resource
            res_locations_list = list(res_locations.keys())
            kd_tree_resources = sp.spatial.KDTree(res_locations_list)
            

            '''
            MAIN INNER LOOP THROUGH ENTITIES
            '''
            # Loop through entities and update their attributes 
            for key, entity in self.entities.ent_dict.items():
                
                # Create a tuple storing location for lookups
                coordinates = (entity.inputs['x_loc'], entity.inputs['y_loc'])
                self.grid[coordinates[1], coordinates[0]]=[
                    np.uint8(round(entity.inputs['R'] + 10, 0)),
                    np.uint8(round(entity.inputs['G']+10, 0)), 
                    np.uint8(round(entity.inputs['B']+10, 0))]

                # Nearest neighbors
                # kd_tree_entities.query returns two items: distances to nearest neighbors, index of each neighbor in self.data (the list that was passed in to form the KD Tree)
                # neighbor_indices: indices of locations of the nearest neighbors in list(ent_locations.keys())

                n_distances, neighbor_indices = kd_tree_entities.query(coordinates, 4) # First coordinate is always self
                entity.inputs['x_nN_1'], entity.inputs['y_nN_1'] = (ent_locations_list[neighbor_indices[1]][0], ent_locations_list[neighbor_indices[1]][1]) if (2 <= len(ent_locations_list)) else (None, None) # If entity has a nearest neighbor, then the location of the nearest neighbor goes here. Else, move toward center
                entity.inputs['x_nN_2'], entity.inputs['y_nN_2'] = (ent_locations_list[neighbor_indices[2]][0], ent_locations_list[neighbor_indices[2]][1]) if (3 <= len(ent_locations_list)) else (None, None)
                entity.inputs['x_nN_3'], entity.inputs['y_nN_3'] = (ent_locations_list[neighbor_indices[3]][0], ent_locations_list[neighbor_indices[3]][1]) if (4 <= len(ent_locations_list)) else (None, None)

               
                # Update values for colors of neighbors
                num_neighbors = 3
                # neighbor_queue: [(x_loc_1, y_loc_1),(x_loc_2, y_loc_2), (x_loc_3, y_loc_3)...]
                # the previous block of code found the three nearest locations. This block of code will collect all the entities at each of those locations and concatentate them into a single queue.
                # This queue will then be queried in the next block in order to find the colors of the nearest neighbors.
                # If there are no nearest neighbors (ie. there is only one entity in the system), then 'neighbor_queue' will return an empty list. 

                neighbor_queue = (
                (list(ent_locations[(entity.inputs['x_nN_1'], entity.inputs['y_nN_1'])]) if ((entity.inputs['x_nN_1'], entity.inputs['y_nN_1']) != (None, None)) else []) +
                (list(ent_locations[(entity.inputs['x_nN_2'], entity.inputs['y_nN_2'])]) if ((entity.inputs['x_nN_2'], entity.inputs['y_nN_2']) != (None, None)) else []) +
                (list(ent_locations[(entity.inputs['x_nN_3'], entity.inputs['y_nN_3'])]) if ((entity.inputs['x_nN_3'], entity.inputs['y_nN_3']) != (None, None)) else [])
                )

                # Go through each of the neighbors in the list and assign to the current entity the colors of its neighbors. 
                for i in range(0, num_neighbors):
                    entity.inputs[f'R_nN_{i+1}'] = self.entities.ent_dict[neighbor_queue[i]].inputs['R'] if (i < len(neighbor_queue)) else 0 # Make dimension zero to treat as normal square
                    entity.inputs[f'G_nN_{i+1}'] = self.entities.ent_dict[neighbor_queue[i]].inputs['G'] if (i < len(neighbor_queue)) else 0
                    entity.inputs[f'B_nN_{i+1}'] = self.entities.ent_dict[neighbor_queue[i]].inputs['B'] if (i < len(neighbor_queue)) else 0

                # Nearest resource calculations
                r_distances, resource_indices = kd_tree_resources.query(coordinates, 3)
                entity.inputs['x_nr_1'], entity.inputs['y_nr_1'] = res_locations_list[resource_indices[0]][0], res_locations_list[resource_indices[0]][1]
                entity.inputs['x_nr_2'], entity.inputs['y_nr_2'] = res_locations_list[resource_indices[1]][0], res_locations_list[resource_indices[1]][1]
                entity.inputs['x_nr_3'], entity.inputs['y_nr_3'] = res_locations_list[resource_indices[2]][0], res_locations_list[resource_indices[2]][1]

                entity.think()
                
                # Handle collisions before updating location                   
                if entity.inputs['id_code'] in collision_events.keys():
                    ent_1, ent_2= entity, self.entities.ent_dict[collision_events[entity.inputs['id_code']]]
                    game = Game(ent_1, ent_2)
#                   # Update resources
                    ent_1.inputs['G'] += game.result[0]
                    ent_2.inputs['G'] += game.result[1]
                    
                    # Update trustworthiness
                    ent_1.inputs['B'] += game.choices[0]
                    ent_2.inputs['B'] += game.choices[1]

                    
                    # Reproduce only if there is space on the grid for another entity
                    if self.num_entities < sim_params['x_dim']*sim_params['y_dim']:
                        #  sim_params, ent_params, id_num, ent_1, ent_2
                        reproduction_results = Reproduction(sim_params, entity_params, 
                                                            ent_1, ent_2, self.num_entities)
                    
                    if reproduction_results.new_ent != None:
                        # add new entities to the dictionary
                        new_ents[reproduction_results.new_ent.inputs['id_code']] = reproduction_results.new_ent
                        # Update number of entities here
                        self.num_entities += 1
                        
                               
                # Update Location
                new_x_loc = entity.inputs['x_loc']+entity.outputs['dx']
                if 0 <= new_x_loc <= sim_params['x_dim']-1:
                    entity.inputs['x_loc'] = new_x_loc

                new_y_loc = entity.inputs['y_loc']+entity.outputs['dy']
                if 0 <= new_y_loc <= sim_params['y_dim']-1:
                    entity.inputs['y_loc'] = new_y_loc
                
                # check for resource collection
                if (entity.inputs['x_loc'], entity.inputs['y_loc']) in res_locations:
                    entity.inputs['G'] += res_locations[(entity.inputs['x_loc'], entity.inputs['y_loc'])].value
                    del res_locations[(entity.inputs['x_loc'], entity.inputs['y_loc'])]
                    
                # Increment age and decrement time to live
                entity.inputs['age'] += 1
                entity.inputs['hp'] -= 1 
                if entity.inputs['hp'] <= 0:
                    # Add it to the list of dead entities at that location (if the list has been initialized)
                    if coordinates not in dead_ents:
                        dead_ents[coordinates] = []
                    if entity.inputs['id_code'] not in dead_ents[coordinates]:
                        dead_ents[coordinates].append(entity.inputs['id_code'])
                
            # Outside of entities loop 
            # Add grid to grids
            self.grids.append(copy.deepcopy(self.grid))

            # Delete dead entities
            for loc, id_codes in dead_ents.items():
                for id_code in id_codes:
                    ent_locations[loc].remove(id_code)
                    del self.entities.ent_dict[id_code]
                if ent_locations[loc] == {}:
                    free_squares.add(loc)

            # Add newly spawned entities back into the entity dictionary for next sim loop
            for id_code, new_ent in new_ents.items():
                
                # Generate locations that are not already in the dictionary
                respawn_loc = random.sample(tuple(free_squares), 1)[0]
                new_ent.inputs['x_loc'], new_ent.inputs['y_loc'] = respawn_loc[0], respawn_loc[1]
                self.entities.ent_dict[id_code] = new_ent
    
    def display_grid(self, grid):
        plt.imshow(grid)

    def animate_simulation(self):
        fig, ax = plt.subplots()
        self.ax = ax
        def update_plot(frame):
            self.display_grid(self.grids[frame])

            ax.set_xticks([])
            ax.set_yticks([])
        
        animation = FuncAnimation(fig, update_plot, frames=len(self.grids), interval=10, repeat=True, repeat_delay=100)
        if self.save_animation == True:
            ffmpeg_path = r"C:\Program Files\FFmpeg\bin\ffmpeg.exe" # Put your path to ffmpeg here
            plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
            mp4_filename = self.animation_name
            animation.save(mp4_filename, writer='ffmpeg', fps=10)
        if self.play_GIF == True:
            plt.show()
        return animation
    