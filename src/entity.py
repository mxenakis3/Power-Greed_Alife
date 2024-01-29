import numpy as np
import scipy as sp
import random

# Creates list of entities
class Entities:
    def __init__(self, sim_params, entity_params):
        # self.ent_dict: contains entity objects, indexed by Id
        self.ent_dict = {}
        
        # self.locations_dict: index is coordinate, value is list of entity ids at that coordinate
        self.locations_dict = {}
        
        for i in range(sim_params['initial_entities']):
            ent = Entity(sim_params, entity_params, id_code = i)
            self.ent_dict[ent.inputs['id_code']] = ent
            coordinate = (ent.inputs['x_loc'], ent.inputs['y_loc'])
            if coordinate not in self.locations_dict:
                self.locations_dict[coordinate] = []
            self.locations_dict[coordinate].append(ent.inputs['id_code'])            
            
    def __str__(self, level=0):
        indent = "    " * level
        result = f"{indent}Entities:\n"
        for key, value in self.ent_dict.items():
            result += f"{indent}  {key}: {value.__str__(level + 1)}"
        return result


# This class is used to generate entities, either on reproduction or on initial setup
# Sets initial values and neural network for newly created creatures
# Use conditional expressions based on reproduction
class Entity:
    def __init__(self, sim_params, entity_params, id_code, reproduction = False, parent_1 = None, parent_2 = None):
        self.inputs = {     # id 
                           'id_code' : id_code,
            
                            # location, ints
                           'x_loc':np.random.randint(0, sim_params['x_dim']-1), 
                           'y_loc':np.random.randint(0, sim_params['y_dim']-1), 
            
                            # Physical fitness - normally distributed from 0-90
                            'R': round(max(10,min(90, sp.stats.norm.rvs(
            loc=((entity_params['max_init_R'] + entity_params['min_init_R']) / 2),
            scale=((entity_params['max_init_R'] - entity_params['min_init_R']) / (2 * sp.stats.norm.ppf(0.995)))))),0) 
            if not reproduction else (parent_1.inputs['R'] + parent_2.inputs['R'] / 2),

                            # Economic Fitness - Uniformally dist~[10,36]
                           'G':np.random.randint(entity_params['min_init_G'], entity_params['max_init_G'])
            if not reproduction else (parent_1.inputs['G'] + parent_2.inputs['G'] / 2),

                            # Initial Trustworthiness
                           'B': round(max(10,min(90, sp.stats.norm.rvs(
            loc=((entity_params['max_init_B'] + entity_params['min_init_B']) / 2),
            scale=((entity_params['max_init_B'] - entity_params['min_init_B']) / (2 * sp.stats.norm.ppf(0.995)))))),0)
            if not reproduction else (parent_1.inputs['B'] + parent_2.inputs['B'] / 2),

                            # Time reamining and age, ints
                            'hp':0, 'age':0, 

                            # Cooperation/ Defection Statistics, int, int, float
                           'num_coop':0, 'num_def':0, 'coop_rate':0,

                            # Locations of top 3 nearest resources ints
                           'x_nr_1':0, 'y_nr_1':0,
                           'x_nr_2':0, 'y_nr_2':0,
                           'x_nr_3':0, 'y_nr_3':0,

                            # Locations of top 3 nearest neighbors, ints
                           'x_nN_1':0, 'y_nN_1':0,
                           'x_nN_2':0, 'y_nN_2':0,
                           'x_nN_3':0, 'y_nN_3':0,

                            # Fitness values of top 3 nearest neighbors, ints
                           'R_nN_1':0, 'G_nN_1':0, 'B_nN_1':0,
                           'R_nN_2':0, 'G_nN_2':0, 'B_nN_2':0,
                           'R_nN_3':0, 'G_nN_3':0, 'B_nN_3':0,
                          }
            
        # Add in some calculated values
        self.inputs['hp'] = min(100, round(((0.5)*self.inputs['R']),0) + self.inputs['G']) 

        # Define a few other parameters
        self.pairing = None
        self.sim_params = sim_params
        self.entity_params = entity_params

        # Initialize outputs, this dictionary will constantly update
        self.outputs = {'dx':0, 'dy':0, 'coop':0, 'reproduce':0}

        # Create weight matrices
        # Cases: 0 hidden layers, 1 hidden layer, 2 hidden layers, any number of hidden layers > 2
        self.neural_net = []
        if reproduction == False:
            weight_lengths = []
            weight_lengths.append(len(self.inputs))
            for length in sim_params['hidden_layers']:
                weight_lengths.append(length)
            weight_lengths.append(len(self.outputs))

            for i in range(len(weight_lengths)-1):
                # def __init__(self, len_input, len_output, p_edge, p_neg_weight, layer=None, reproduction=False)
                matrix = Weight_Matrix(len_input=weight_lengths[i], len_output=weight_lengths[i+1], p_edge=self.entity_params['p_edge'], p_neg_weight=self.entity_params['p_neg_weight'])
                self.neural_net.append(matrix)
        else:
            p1_nn = parent_1.neural_net
            p2_nn = parent_2.neural_net
            for i in range(len(p1_nn)):
                p1_layer = p1_nn[i].weights
                p2_layer = p2_nn[i].weights
                curr_layer = np.empty_like(p1_layer)
                for index, _ in np.ndenumerate(p1_layer):
                    # print(f"index: {index}")
                    i, j = index
                    if (i+j) % 2 == 0:
                        curr_layer[index] = p1_layer[index]
                    else:
                        curr_layer[index] = p2_layer[index]
                curr_layer_weight_matrix = Weight_Matrix(layer=curr_layer, reproduction=True) # Need to 'wrap' the current layer as a weight object
                self.neural_net.append(curr_layer_weight_matrix)
            
            
    def __str__(self, level=0):
        indent = "    " * level
        result = f"{indent}Entity:\n"
        for key, value in self.inputs.items():
            if isinstance(value, Entity):
                result += f"{indent}  {key}: {value.__str__(level + 1)}"
            else:
                result += f"{indent}  {key}: {value}\n"
        return result

    def think(self): # Updates self.outputs given inputs
        # Normalize inputs (convert x_loc to float, etc.)
        self.inputs['x_nN_1'] = self.sim_params['x_dim'] // 2 if self.inputs['x_nN_1'] == None else self.inputs['x_nN_1'] 
        self.inputs['y_nN_1'] = self.sim_params['y_dim'] // 2 if self.inputs['y_nN_1'] == None else self.inputs['y_nN_1'] 

        self.inputs['x_nN_2'] = self.sim_params['x_dim'] // 2 if self.inputs['x_nN_2'] == None else self.inputs['x_nN_2'] 
        self.inputs['y_nN_2'] = self.sim_params['y_dim'] // 2 if self.inputs['y_nN_2'] == None else self.inputs['y_nN_2'] 

        self.inputs['x_nN_3'] = self.sim_params['x_dim'] // 2 if self.inputs['x_nN_3'] == None else self.inputs['x_nN_3'] 
        self.inputs['y_nN_3'] = self.sim_params['y_dim'] // 2 if self.inputs['y_nN_3'] == None else self.inputs['y_nN_3'] 

        inputs = np.array(list(self.inputs.values()))
        norm_vector = np.array([0, 1/self.sim_params['x_dim'], 1/self.sim_params['y_dim'],
                      1/225, 1/225, 1/225,
                      1/self.entity_params['max_age'], 1/self.entity_params['max_age'],
                      0, 0, 1,
                      1/self.sim_params['x_dim'], 1/self.sim_params['y_dim'],
                      1/self.sim_params['x_dim'], 1/self.sim_params['y_dim'],
                      1/self.sim_params['x_dim'], 1/self.sim_params['y_dim'],
                      1/self.sim_params['x_dim'], 1/self.sim_params['y_dim'],
                      1/self.sim_params['x_dim'], 1/self.sim_params['y_dim'],
                      1/self.sim_params['x_dim'], 1/self.sim_params['y_dim'],
                      1/225, 1/225, 1/225,
                      1/225, 1/225, 1/225,
                      1/225, 1/225, 1/225])
        
        norm_inputs = np.multiply(inputs, norm_vector) # Gets normalized inputs as an nx1 array

        # Initialize the input layer
        output_vals = norm_inputs

        # Calculate the output for each layer
        for i in range(len(self.neural_net)):
            output_vals = np.tanh(np.dot(output_vals, self.neural_net[i].weights))

        # Round outputs to the nearest whole number
        output_vals = [round(x) for x in output_vals]

        # Update outputs dictionary with the values from think
        for key, value in zip(self.outputs, output_vals):
            self.outputs[key] = value

        # For temporary printing purposes
        return output_vals
    

# This class is used to generate the initial set of neural networks that the entities will be using
class Weight_Matrix:
    def __init__(self, len_input=None, len_output=None, p_edge=None, p_neg_weight=None, layer=None, reproduction=False):
        if reproduction == False:
            self.weights = np.empty((len_input, len_output)) 
            for i in range(self.weights.shape[0]):
                for j in range(self.weights.shape[1]):
                    if random.random() <= p_edge and i != j: # No self loops
                        sign = -1 if random.random() <= p_neg_weight else 1 # Make edge weight negative with prob. p_neg_weight
                        self.weights[i,j] = sign*random.uniform(0,4) # Create an edge weight between -4 and 4 
                    else:
                        self.weights[i,j] = 0
        else:
            self.weights = layer
                    
    def __str__(self):
        return f"{self.weights}"


#  Returns an object containing the results of a game between two players
class Game:
    def __init__(self, ent_1, ent_2):
        # Chips to play with
#         print(f"Game occurs between {ent_1.inputs['id_code']} & {ent_2.inputs['id_code']}")
        
        # Decision to cooperate/ Defect
        ent_1_choice = ent_1.outputs['coop'] # {1, -1, 0}
        ent_2_choice = ent_2.outputs['coop']
        
        outcomes = {
            (1, 1): (7, 7), # Both players cooperate 
            (1, -1): (0, 10), # ent_1 cooperates, ent_2 defects
            (-1, 1): (10, 0), # ent_1 defects, ent_2 cooperates
            (-1, -1): (0,0) # Both players defect
        }
        
        self.result = outcomes.get((ent_1_choice, ent_2_choice), (0,0))
        
        # Choices present only if the game is played
        self.choices = (0, 0)
        if self.result != (0,0):
                self.choices = (ent_1_choice, ent_2_choice)
    
    def __str__(self):
        return f"{self.result}"
        
# Returns an entity if reproduction occurs, returns nothing if not
class Reproduction:
    def __init__(self, sim_params, ent_params, ent_1, ent_2, num_entities):
        
        ent_1_choice, ent_2_choice = ent_1.outputs['reproduce'], ent_2.outputs['reproduce']
#         print(f"reproduction occurs between {ent_1.inputs['id_code']} & {ent_2.inputs['id_code']}")
        outcomes = {
            (1, 1): Entity(sim_params=sim_params, entity_params=ent_params,
                            id_code= num_entities+1, reproduction=True, parent_1=ent_1, parent_2=ent_2),
            # Both players wish to reproduce
            (1, 0): Entity(sim_params=sim_params, entity_params=ent_params,
                            id_code=num_entities+1, reproduction=True, parent_1=ent_1, parent_2=ent_2),
            # ent_2 will reproduce if ent_1 is down
            (0, 1): Entity(sim_params=sim_params, entity_params=ent_params,
                            id_code= num_entities+1, reproduction=True, parent_1=ent_1, parent_2=ent_2),
            # ent_1 will reproduce if ent_2 is down
        }
        self.new_ent = outcomes.get((ent_1_choice, ent_2_choice), None)