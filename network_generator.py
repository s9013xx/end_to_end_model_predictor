import math
import numpy as np
import pandas as pd
import os
import argparse
# from operator import itemgetter
# import cnn
import state_enumerator as se
from state_string_utils import StateStringUtils
import state_space_parameters as cfg

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
        parameters.output_model_filename = 'network' + '_' + parameters.min_layer + '_' + parameters.max_layer + '_' + parameters.network_number + '.csv'
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
    def __init__(self, minimum_layer, maximum_layer, network_number, qstore=None):
        self.state_list = []
        self.state_space_parameters = state_space_parameters
        # Class that will expand states for us
        self.enum = se.StateEnumerator(state_space_parameters)
        self.stringutils = StateStringUtils(state_space_parameters, 0, 0, 0)
        # Starting State
        self.state = se.State('start', 0, 1, 0, 0, state_space_parameters.image_size, 0, 0, 0, 0, 0, 0, 0, 0, 0)# if not state else state
        self.bucketed_state = self.enum.bucket_state(self.state)

        # Cached Q-Values -- used for q learning update and transition
        self.qstore = QValues() if not qstore else qstore
        # self.replay_dictionary = replay_dictionary

        # self.epsilon=epsilon # epsilon: parameter for epsilon greedy strategy
        self.network_number = network_number

    def update_replay_database(self, new_replay_dic):
        self.replay_dictionary = new_replay_dic

    def generate_net(self):
        # Have Q-Learning agent sample current policy to generate a network and convert network to string format
        self._reset_for_new_walk()
        state_list = self._run_agent()
        # state_list = self.stringutils.add_drop_out_states(state_list)
        net_string = self.stringutils.state_list_to_string(state_list)

        return net_string
        # Check if we have already trained this model
        # if net_string in self.replay_dictionary['net'].values:
        #     acc_best_val = self.replay_dictionary[self.replay_dictionary['net']==net_string]['accuracy_best_val'].values[0]
        #     iter_best_val = self.replay_dictionary[self.replay_dictionary['net']==net_string]['iter_best_val'].values[0]
        #     acc_last_val = self.replay_dictionary[self.replay_dictionary['net']==net_string]['accuracy_last_val'].values[0]
        #     iter_last_val = self.replay_dictionary[self.replay_dictionary['net']==net_string]['iter_last_val'].values[0]
        #     acc_best_test = self.replay_dictionary[self.replay_dictionary['net']==net_string]['accuracy_best_test'].values[0]
        #     acc_last_test = self.replay_dictionary[self.replay_dictionary['net']==net_string]['accuracy_last_test'].values[0]
        #     machine_run_on = self.replay_dictionary[self.replay_dictionary['net']==net_string]['machine_run_on'].values[0]
        # else:
        #     acc_best_val = -1.0
        #     iter_best_val = -1.0
        #     acc_last_val = -1.0
        #     iter_last_val = -1.0
        #     acc_best_test = -1.0
        #     acc_last_test = -1.0
        #     machine_run_on = -1.0

        # return (net_string, acc_best_val, iter_best_val, acc_last_val, iter_last_val, acc_best_test, acc_last_test, machine_run_on)

    def _reset_for_new_walk(self):
        '''Reset the state for a new random walk'''
        # Architecture String
        self.state_list = []

        # Starting State
        self.state = se.State('start', 0, 1, 0, 0, self.state_space_parameters.image_size, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.bucketed_state = self.enum.bucket_state(self.state)

    def _run_agent(self):
        ''' Have Q-Learning agent sample current policy to generate a network
        '''
        while self.state.terminate == 0:
            self._transition_q_learning()

        return self.state_list

    def _transition_q_learning(self):
        ''' Updates self.state according to an epsilon-greedy strategy'''
        if self.bucketed_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(self.bucketed_state, self.qstore.q)

        action_values = self.qstore.q[self.bucketed_state.as_tuple()]

        # print action_values

        action = se.State(state_list=action_values['actions'][np.random.randint(len(action_values['actions']))])
        # epsilon greedy choice
        # if np.random.random() < self.epsilon:
        #     action = se.State(state_list=action_values['actions'][np.random.randint(len(action_values['actions']))])
        # else:
        #     max_q_value = max(action_values['utilities'])
        #     max_q_indexes = [i for i in range(len(action_values['actions'])) if action_values['utilities'][i]==max_q_value]
        #     max_actions = [action_values['actions'][i] for i in max_q_indexes]
        #     action = se.State(state_list=max_actions[np.random.randint(len(max_actions))])

        self.state = self.enum.state_action_transition(self.state, action)
        self.bucketed_state = self.enum.bucket_state(self.state)

        self._post_transition_updates()

    def _post_transition_updates(self):
        #State to go in state list
        bucketed_state = self.bucketed_state.copy()

        self.state_list.append(bucketed_state)

    def sample_replay_for_update(self):
        # Experience replay to update Q-Values
        for i in range(self.state_space_parameters.replay_number):
            net = np.random.choice(self.replay_dictionary['net'])
            accuracy_best_val = self.replay_dictionary[self.replay_dictionary['net'] == net]['accuracy_best_val'].values[0]
            accuracy_last_val = self.replay_dictionary[self.replay_dictionary['net'] == net]['accuracy_last_val'].values[0]
            state_list = self.stringutils.convert_model_string_to_states(cnn.parse('net', net))

            state_list = self.stringutils.remove_drop_out_states(state_list)

            # Convert States so they are bucketed
            state_list = [self.enum.bucket_state(state) for state in state_list]

            self.update_q_value_sequence(state_list, self.accuracy_to_reward(accuracy_best_val))


def main():
    parameters = read_network_generator_parameters()
    parameters = complete_file_path(parameters)
    read_network_generator_parameters()
    print(parameters)
    auto_create_dir(parameters)

    net_list = []
    for i in range(parameters.network_number):
      print('generate : ', i)
      network_generator = Network_Generator(parameters.minimum_layer, parameters.maximum_layer, parameters.network_number)
      net_list.append(network_generator.generate_net())
      print('net_list:', net_list)
    # df_net = pd.DataFrame(net_list, columns=['network'])
    # df_net.to_csv(parameters.output_model_path,index=False)

if __name__ == '__main__':
    main()




