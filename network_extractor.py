import math
# import numpy as np
import pandas as pd
import os
import argparse
from termcolor import colored
from state_string_utils import StateStringUtils
import argparse
import subprocess
import cnn

def read_network_extractor_parameters():
    parser = argparse.ArgumentParser('Network Executor')
    # Model Parameters
    parser.add_argument('--network_number', '-net_num', type=int, default=10000, help='Generate the number of end to end models')
    parser.add_argument('--min_layer', '-min_layer', type=int, default=1, help='Generate the minumum number of end to end models')
    parser.add_argument('--max_layer', '-max_layer', type=int, default=25, help='Generate the maximum number of end to end models')
    # General Parameters
    parser.add_argument('--default_dirname', '-dd', type=str, default='data', help='The dirname of the output csv filename in generate model step')
    parser.add_argument('--cpu', action="store_true", default=False, help='Benchmark using CPU')
    # Input File Parameters
    parser.add_argument('--input_model_exe_dirname', '-imed', type=str, default='model_exe', help='The dirname of the output csv filename in generate model step')
    parser.add_argument('--input_model_exe_filename', '-imef', type=str, default='', help='The output csv file name')
    parser.add_argument('--input_model_exe_path', '-imep', type=str, default='', help='The path of the output csv filename in generate model step')
    # Output File Parameters
    parser.add_argument('--output_model_feature_dirname', '-omfd', type=str, default='model_feature', help='The dirname of the output csv filename in generate model step')
    parser.add_argument('--output_model_feature_filename', '-omff', type=str, default='', help='The output csv file name')
    parser.add_argument('--output_model_feature_path', '-omfp', type=str, default='', help='The path of the output csv filename in generate model step')
    # Parse Arguments
    args = parser.parse_args()
    return args

def complete_file_path(parameters):
    if not parameters.input_model_exe_filename:
        parameters.input_model_exe_filename = 'network' + '_' + str(parameters.min_layer) + '_' + str(parameters.max_layer) + '_' + str(parameters.network_number) + '.csv'
    if not parameters.input_model_exe_path:
        parameters.input_model_exe_path = os.path.join(os.getcwd(), parameters.default_dirname, parameters.input_model_exe_dirname, parameters.input_model_exe_filename)
    
    if not parameters.output_model_feature_filename:
        parameters.output_model_feature_filename = 'network' + '_' + str(parameters.min_layer) + '_' + str(parameters.max_layer) + '_' + str(parameters.network_number) + '.csv'
    if not parameters.output_model_feature_path:
        parameters.output_model_feature_path = os.path.join(os.getcwd(), parameters.default_dirname, parameters.output_model_feature_dirname, parameters.output_model_feature_filename)
    
    return parameters

def auto_create_dir(parameters):
    def create_dir_elemenet(path):
        warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
        if not os.path.isdir(path):
            os.makedirs(path)
            print(warn_tag + 'Auto create dir: ' + path)
    create_dir_elemenet(parameters.default_dirname)
    create_dir_elemenet(os.path.dirname(parameters.output_model_feature_path))
    return

def main():
    parameters = read_network_extractor_parameters()
    parameters = complete_file_path(parameters)
    print(parameters)
    auto_create_dir(parameters)

    df_net = pd.read_csv(parameters.input_model_exe_path)

    for i in range(df_net.shape[0]):
        if i%1000==0:
            print("=====>", i)
        # print(df_net.iloc[i]['net_string'])
        net_list = cnn.parse('net', df_net.iloc[i]['net_string'])
        network_list = []
        is_contain_fc = 0
        layer_count = 0

        for layer in net_list:
            # print 'layer : ', layer
            if layer[0]=='input':
                network_list.append([1,0,0,0,0,
                    layer[1], layer[2]**2, layer[3],
                    0,0,0,0,0,0,
                    0,0,0,
                    0,0,0,
                    0,0,0])
            elif layer[0]=='conv':
                layer_count = layer_count + 1
                network_list.append([0,1,0,0,0,
                    layer[1], layer[2]**2, layer[3],
                    layer[4],layer[5],layer[6],layer[7],layer[8],layer[9],
                    0,0,0,
                    0,0,0,
                    0,0,0])
            elif layer[0]=='pool':
                layer_count = layer_count + 1
                network_list.append([0,0,1,0,0,
                    layer[1], layer[2]**2, layer[3],
                    0,0,0,0,0,0,
                    layer[4],layer[5],layer[6],
                    0,0,0,
                    0,0,0])
            elif layer[0]=='fc':
                is_contain_fc = 1
                layer_count = layer_count + 1
                network_list.append([0,0,0,1,0,
                    layer[1], layer[2], layer[3],
                    0,0,0,0,0,0,
                    0,0,0,
                    layer[4],layer[5],layer[6],
                    0,0,0])
            elif layer[0]=='output':
                if is_contain_fc == 0:
                    output = layer[2]**2
                else:
                    output = layer[2]
                network_list.append([0,0,0,0,1,
                                      0,0,0,
                                      0,0,0,0,0,0,
                                      0,0,0,
                                      0,0,0,
                                      layer[1], output, layer[3]])
        
        

        for _ in range(parameters.max_layer-layer_count):
            network_list.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        df_net.at[i, 'network_list'] = str(network_list)
    # print(df_net)
    df_net.to_csv(parameters.output_model_feature_path)
    # layer type ->   0:input        1:conv        2:pool       3:fc        4:output
    # input ->        5:batch_size   6:input_image_size**2      7:input_channels
    # convolution ->  8:filters      9:kernelsize  10:strides   11:padding    12:activation_fct   13:use_bias
    # pooling ->      14:poolsize    15:strides    16:padding
    # fc ->           17:units       18:activation_fct   19:use_bias
    # output ->       20:batch_size   21:output_image_size**2      22:output_channels

    '''Defines all state transitions, populates q_values where actions are valid
    conv[0] -> 'conv'
    conv[1] -> filters
    conv[2] -> kernal size
    conv[3] -> strides
    conv[4] -> padding
    conv[5] -> activation
    conv[6] -> bias

    pool[0] -> 'pool'
    pool[1] -> pool size
    pool[2] -> stride
    pool[3] -> padding

    fc[0] -> 'fc'
    fc[1] -> units
    fc[2] -> activation
    fc[3] -> bias

   '''

if __name__ == '__main__':
    main()




