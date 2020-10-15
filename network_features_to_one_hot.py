import copy
from abc import ABCMeta, abstractmethod
import argparse
from termcolor import colored
import numpy as np
import pandas as pd
import json
import ast
import os

def read_network_features_to_one_hot_parameters():
    parser = argparse.ArgumentParser('Network Executor')
    # Model Parameters
    parser.add_argument('--network_number', '-net_num', type=int, default=10000, help='Generate the number of end to end models')
    parser.add_argument('--min_layer', '-min_layer', type=int, default=1, help='Generate the minumum number of end to end models')
    parser.add_argument('--max_layer', '-max_layer', type=int, default=25, help='Generate the maximum number of end to end models')
    # General Parameters
    parser.add_argument('--default_dirname', '-dd', type=str, default='data', help='The dirname of the output csv filename in generate model step')
    parser.add_argument('--cpu', action="store_true", default=False, help='Benchmark using CPU')
    # Input File Parameters
    parser.add_argument('--input_model_feature_dirname', '-imfd', type=str, default='model_feature', help='The dirname of the output csv filename in generate model step')
    parser.add_argument('--input_model_feature_filename', '-imff', type=str, default='', help='The output csv file name')
    parser.add_argument('--input_model_feature_path', '-imfp', type=str, default='', help='The path of the output csv filename in generate model step')
    # Output File Parameters
    parser.add_argument('--output_model_one_hot_dirname', '-omohd', type=str, default='model_one_hot', help='The dirname of the output csv filename in generate model step')
    parser.add_argument('--output_model_one_hot_filename', '-omohf', type=str, default='', help='The output csv file name')
    parser.add_argument('--output_model_one_hot_path', '-omohp', type=str, default='', help='The path of the output csv filename in generate model step')
    # Parse Arguments
    args = parser.parse_args()
    return args

def complete_file_path(parameters):
    if not parameters.input_model_feature_filename:
        parameters.input_model_feature_filename = 'network' + '_' + str(parameters.min_layer) + '_' + str(parameters.max_layer) + '_' + str(parameters.network_number) + '.csv'
    if not parameters.input_model_feature_path:
        parameters.input_model_feature_path = os.path.join(os.getcwd(), parameters.default_dirname, parameters.input_model_feature_dirname, parameters.input_model_feature_filename)
    
    if not parameters.output_model_one_hot_filename:
        parameters.output_model_one_hot_filename = 'network' + '_' + str(parameters.min_layer) + '_' + str(parameters.max_layer) + '_' + str(parameters.network_number) + '.csv'
    if not parameters.output_model_one_hot_path:
        parameters.output_model_one_hot_path = os.path.join(os.getcwd(), parameters.default_dirname, parameters.output_model_one_hot_dirname, parameters.output_model_one_hot_filename)
    
    return parameters

def auto_create_dir(parameters):
    def create_dir_elemenet(path):
        warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
        if not os.path.isdir(path):
            os.makedirs(path)
            print(warn_tag + 'Auto create dir: ' + path)
    create_dir_elemenet(parameters.default_dirname)
    create_dir_elemenet(os.path.dirname(parameters.output_model_one_hot_path))
    return


def main():
    parameters = read_network_features_to_one_hot_parameters()
    parameters = complete_file_path(parameters)
    print(parameters)
    auto_create_dir(parameters)

    df_row  = pd.read_csv(parameters.input_model_feature_path)

    column_string = list()
    network_list = eval(df_row.loc[0, 'network_list'])
    column_count = 0
    for layer in network_list:
        # print(layer)
        for cell in layer:
            # print(cell)
            column_count = column_count+1
    # print column_count
    for i in range(column_count):
        column_string.append(str(i))
    column_string.append('latency')

    oh_df = pd.DataFrame()

    for index in range(df_row.shape[0]):
        if index%1000==0:
            print("=====>", index)
        network_list = eval(df_row.loc[index, 'network_list'])
        # print network_list
        latency = df_row.loc[index, 'time_trim_mean']

        np_all = np.array([])
        for layer in network_list:
            for cell in layer:
                np_all = np.append(np_all, [cell])
        np_all = np.append(np_all, [float(latency)])
        # print(np_all)
        

        one_df = pd.DataFrame([np_all], columns=column_string)
        oh_df = oh_df.append(one_df)

    oh_df.to_csv(parameters.output_model_one_hot_path)
    # if not os.path.isfile(file_path):
    #     oh_df.to_csv(file_path, header=column_string)
    # else:
    #     oh_df.to_csv(file_path, mode='a', header=False)

if __name__ == '__main__':
    main()
