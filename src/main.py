import numpy as np
import random
import imageio


from simulation import Simulation

np.random.seed(123)
random.seed(1)

def main():
    print("simulation begins")

    sim_params = {
        # 'frames':30,
        'iterations':100,
        'initial_entities':750,
        'x_dim': 200,
        'y_dim': 150,
        'res_spawn_rate':3,
        # 'res_lifespan':7,
        # number of internal neurons in each of the hidden layers
        'hidden_layers': [11,7],
    }
    
    entity_params = {
        'internal_nodes': 30,
        'p_edge':.4,
        'p_neg_weight':.5,
        'max_age':100,
        'min_init_R':10,
        'max_init_R':90,
        'min_init_G': 10,
        'max_init_G': 36,
        'min_init_B': 0,
        'max_init_B': 100,
    }  
    sim = Simulation(sim_params, entity_params, save_animation=True, animation_name='200_150_750_100.mp4', play_GIF=False)
    sim.animate_simulation()
    
main()