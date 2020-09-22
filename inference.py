import os
import argparse
import tensorflow as tf
from scipy import stats
from state_string_utils import StateStringUtils
import cnn

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
parser = argparse.ArgumentParser('Collect Actual Data Parser')
# Network parameters
parser.add_argument('--net_string', type=str, default='', help='netwok list')
# General parameters
parser.add_argument('--iter_warmup', type=int, default=5, help='Number of iterations for warm-up')
parser.add_argument('--iter_benchmark', type=int, default=10, help='Number of iterations for benchmark')
parser.add_argument('--cpu', action="store_true", default=False, help='Benchmark using CPU')
args = parser.parse_args()


if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main(_):
    ########### Benchmark end-to-end network ##########
    net_list = cnn.parse('net', args.net_string)
    # print(net_list)
    # exit()
    StateStringUtils().convert_model_string_to_states(net_list)

if __name__ == '__main__':
    tf.app.run()
