import math
# import numpy as np
import pandas as pd
import os
import argparse
from termcolor import colored
from state_string_utils import StateStringUtils
import argparse
import subprocess

def read_network_executor_parameters():
    parser = argparse.ArgumentParser('Network Executor')
    # Model Parameters
    parser.add_argument('--network_number', '-net_num', type=int, default=10000, help='Generate the number of end to end models')
    parser.add_argument('--min_layer', '-min_layer', type=int, default=1, help='Generate the minumum number of end to end models')
    parser.add_argument('--max_layer', '-max_layer', type=int, default=25, help='Generate the maximum number of end to end models')
    # General Parameters
    parser.add_argument('--default_dirname', '-dd', type=str, default='data', help='The dirname of the output csv filename in generate model step')
    parser.add_argument('--cpu', action="store_true", default=False, help='Benchmark using CPU')
    # Input File Parameters
    parser.add_argument('--input_model_dirname', '-imd', type=str, default='model_csv', help='The dirname of the output csv filename in generate model step')
    parser.add_argument('--input_model_filename', '-imf', type=str, default='', help='The output csv file name')
    parser.add_argument('--input_model_path', '-imp', type=str, default='', help='The path of the output csv filename in generate model step')
    # Output File Parameters
    parser.add_argument('--output_model_exe_dirname', '-omed', type=str, default='model_exe', help='The dirname of the output csv filename in generate model step')
    parser.add_argument('--output_model_exe_filename', '-omef', type=str, default='', help='The output csv file name')
    parser.add_argument('--output_model_exe_path', '-omep', type=str, default='', help='The path of the output csv filename in generate model step')
    # Parse Arguments
    args = parser.parse_args()
    return args

def complete_file_path(parameters):
    if not parameters.input_model_filename:
        parameters.input_model_filename = 'network' + '_' + str(parameters.min_layer) + '_' + str(parameters.max_layer) + '_' + str(parameters.network_number) + '.csv'
    if not parameters.input_model_path:
        parameters.input_model_path = os.path.join(os.getcwd(), parameters.default_dirname, parameters.input_model_dirname, parameters.input_model_filename)
    
    if not parameters.output_model_exe_filename:
        parameters.output_model_exe_filename = 'network' + '_' + str(parameters.min_layer) + '_' + str(parameters.max_layer) + '_' + str(parameters.network_number) + '.csv'
    if not parameters.output_model_exe_path:
        parameters.output_model_exe_path = os.path.join(os.getcwd(), parameters.default_dirname, parameters.output_model_exe_dirname, parameters.output_model_exe_filename)
    
    return parameters

def auto_create_dir(parameters):
    def create_dir_elemenet(path):
        warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
        if not os.path.isdir(path):
            os.makedirs(path)
            print(warn_tag + 'Auto create dir: ' + path)
    create_dir_elemenet(parameters.default_dirname)
    create_dir_elemenet(os.path.dirname(parameters.output_model_exe_path))
    return

def main():
    parameters = read_network_executor_parameters()
    parameters = complete_file_path(parameters)
    print(parameters)
    auto_create_dir(parameters)

    df_net = pd.read_csv(parameters.input_model_path)

    created_file = 0
    count = 0
    for net_string in df_net['network']:
      # count = count + 1
      # if count > 1:
      #   exit()
      print('net_string: ', net_string)
      if parameters.cpu:
        command = 'python inference.py --cpu --net_string="%s" > temp' % net_string
      else:
        command = 'python inference.py --net_string="%s" > temp' % net_string
      print("command: ", command)
      process = subprocess.Popen(command, stdout = subprocess.PIPE, shell = True)
      process.wait()

      time_max = None
      time_min = None
      time_median = None
      time_mean = None
      time_trim_mean = None

      line_count = 0
      tmp_file = open("temp", "r")
      if os.stat("temp").st_size == 0:
        continue
      # line = None
      time_data_ele = None
      for line in tmp_file:
          # print line
          time_data_ele = eval(line)
      time_data_ele.update( {'net_string' : net_string} )
       
      df_ele = pd.DataFrame(data = time_data_ele, index=[0])
      print("time_mean: {} ms".format(time_data_ele['time_mean']))

      if created_file==0: 
          df_ele.to_csv(parameters.output_model_exe_path, index=False)
          created_file = 1
      else:
          df_ele.to_csv(parameters.output_model_exe_path, index=False, mode='a', header=False)

if __name__ == '__main__':
    main()  




