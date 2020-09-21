import math
import numpy as np
import pandas as pd
import os
import argparse
from termcolor import colored
# from operator import itemgetter
# import cnn
import random
import state_enumerator as se
from state_string_utils import StateStringUtils
import state_space_parameters

def read_network_generator_parameters():
    parser = argparse.ArgumentParser('Network Generator')
    # Model Parameters
    parser.add_argument('--network_number', '-net_num', type=int, default=10000, help='Generate the number of end to end models')
    parser.add_argument('--min_layer', '-min_layer', type=int, default=1, help='Generate the minumum number of end to end models')
    parser.add_argument('--max_layer', '-max_layer', type=int, default=25, help='Generate the maximum number of end to end models')
    # Generate Parameters
    parser.add_argument('--default_dirname', '-dd', type=str, default='data', help='The dirname of the output csv filename in generate model step')
    parser.add_argument('--output_model_dirname', '-omd', type=str, default='model_csv', help='The dirname of the output csv filename in generate model step')
    parser.add_argument('--output_model_filename', '-omf', type=str, default='', help='The output csv file name')
    parser.add_argument('--output_model_path', '-omp', type=str, default='', help='The path of the output csv filename in generate model step')
    # Parse Arguments
    args = parser.parse_args()
    return args

def complete_file_path(parameters):
    if not parameters.output_model_filename:
        parameters.output_model_filename = 'network' + '_' + str(parameters.min_layer) + '_' + str(parameters.max_layer) + '_' + str(parameters.network_number) + '.csv'
    if not parameters.output_model_path:
        parameters.output_model_path = os.path.join(os.getcwd(), parameters.default_dirname, parameters.output_model_dirname, parameters.output_model_filename)

    return parameters

def auto_create_dir(parameters):
    def create_dir_elemenet(path):
        warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
        if not os.path.isdir(path):
            os.makedirs(path)
            print(warn_tag + 'Auto create dir: ' + path)
    create_dir_elemenet(parameters.default_dirname)
    create_dir_elemenet(os.path.dirname(parameters.output_model_path))
    return

class Network_Generator:
    def __init__(self, state_space_parameters, min_layer, max_layer, qstore=None):
        self.state_list = []
        self.ssp = state_space_parameters
        # Class that will expand states for us
        self.enum = se.StateEnumerator(state_space_parameters, min_layer, max_layer)
        self.stringutils = StateStringUtils()
        # Starting State
        batch_size = self.ssp.possible_batch_size[np.random.randint(len(self.ssp.possible_batch_size))]
        input_image_size = random.randint(1, self.ssp.max_input_size)
        input_channel = self.ssp.possible_input_channel[np.random.randint(len(self.ssp.possible_input_channel))]
        self.previous_state = se.State('start', 0, batch_size, input_image_size, input_channel, 0, 0, 0, 0, 0, 0, 0, 0, input_image_size, input_channel)

    def generate_net(self):
        state_list = self._run_agent()
        net_string = self.stringutils.state_list_to_string(state_list)
        return net_string

    def _run_agent(self):
        while self.previous_state.terminate == 0:
            current_state = self.enum.enumerate_state(self.previous_state)
            self.state_list.append(current_state)
            self.previous_state = current_state
        return self.state_list

def main():
    parameters = read_network_generator_parameters()
    parameters = complete_file_path(parameters)
    read_network_generator_parameters()
    print(parameters)
    auto_create_dir(parameters)

    net_list = []
    for i in range(parameters.network_number):
      network_generator = Network_Generator(state_space_parameters, parameters.min_layer, parameters.max_layer)
      network = network_generator.generate_net()
      print('generate : ', i, ', network:', network)
      net_list.append(network)
      
    # df_net = pd.DataFrame(net_list, columns=['network'])
    # df_net.to_csv(parameters.output_model_path,index=False)

if __name__ == '__main__':
    main()




