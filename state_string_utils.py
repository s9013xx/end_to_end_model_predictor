import math
import numpy as np
from operator import itemgetter
import tensorflow as tf
import state_enumerator as se
import random
import time
import numpy
from scipy import stats
import json

class StateStringUtils:
    ''' Contains all functions dealing with converting nets to net strings
        and net strings to state lists.
    '''
    def __init__(self):
        pass

    def state_list_to_string(self, state_list):
        '''Convert the list of strings to a string we can train from according to the grammar'''
        out_string = ''
        strings = []
        print("len(state_list):", len(state_list))
        i = 0
        while i < len(state_list):
            state = state_list[i]
            if self.state_to_string(state):
                strings.append(self.state_to_string(state))
            i += 1
        return str('[' + ', '.join(strings) + ']')

    def state_to_string(self, state):
        ''' Returns the string asociated with state.
        '''
        if state.layer_type == 'start':
            return 'IN(%i,%i,%i)' % (state.batch_size, state.input_image_size, state.input_channel)
        elif state.terminate == 1:
            return 'OUT(%i,%i,%i)' % (state.batch_size, state.output_image_size, state.output_channel)
        elif state.layer_type == 'conv':
            return 'C(%i,%i,%i,%i,%i,%i,%i,%i,%i)' % (state.batch_size, state.input_image_size, state.input_channel, state.filter_depth, state.filter_size, state.stride, state.padding, state.act, state.bias)
        elif state.layer_type == 'pool':
            return 'P(%i,%i,%i,%i,%i,%i)' % (state.batch_size, state.input_image_size, state.input_channel, state.filter_size, state.stride, state.padding)
        elif state.layer_type == 'fc':
            return 'FC(%i,%i,%i,%i,%i,%i)' % (state.batch_size, state.input_image_size, state.input_channel, state.fc_size, state.act, state.bias)
        return None

    def convert_model_string_to_states(self, parsed_list, start_state=None):
        '''Takes a parsed model string and returns a recursive list of states.'''
        activation_list = ['None', 'tf.nn.relu']

        first_layer = 1
        input_ = None
        tf_placeholder  = None
        op = None
        layer_count = 0
        
        total_conv_filters = 0
        total_conv_kernelsizes = 0
        total_conv_strides = 0
        total_conv_paddings = 0
        total_conv_acts = 0
        total_conv_bias = 0

        total_pool_sizes = 0
        total_pool_strides = 0
        total_pool_paddings = 0

        total_fc_units = 0
        total_fc_acts = 0
        total_fc_bias = 0

        time_list = []
        time_max = None
        time_min = None
        time_median = None
        time_mean = None
        time_trim_mean = None

        tf.reset_default_graph()
        for layer in parsed_list:
            if layer[0] == 'input':
                batchsize = layer[1]
                input_image_size = layer[2]
                input_image_channels = layer[3]
                input_ = np.random.normal(127, 60, (layer[1], layer[2], layer[2], layer[3])).astype(float)
                tf_placeholder = tf.placeholder(tf.float32, shape=[None, layer[2], layer[2], layer[3]], name="inputs")
            if layer[0] == 'conv':
                if first_layer == 1:
                    first_layer = 0
                    op = tf.layers.conv2d(tf_placeholder, filters=layer[4], kernel_size=[layer[5], layer[5]], strides=(layer[6], layer[6]), 
                        padding=('SAME' if layer[7] ==1 else 'VALID'), activation=eval(activation_list[layer[8]]), 
                        use_bias=layer[9], name='convolution_%d'%(layer_count))
                else:
                    op = tf.layers.conv2d(op, filters=layer[4], kernel_size=[layer[5], layer[5]], strides=(layer[6], layer[6]), 
                        padding=('SAME' if layer[7] ==1 else 'VALID'), activation=eval(activation_list[layer[8]]), 
                        use_bias=layer[9], name='convolution_%d'%(layer_count))

                total_conv_filters = total_conv_filters + layer[4]
                total_conv_kernelsizes = total_conv_kernelsizes + layer[5]**2
                total_conv_strides = total_conv_strides + layer[6]**2
                total_conv_paddings = total_conv_paddings + layer[7]
                total_conv_acts = total_conv_acts + layer[8]
                total_conv_bias = total_conv_bias + layer[9]

            elif layer[0] == 'pool':
                if first_layer == 1:
                    first_layer = 0
                    op = tf.layers.max_pooling2d(tf_placeholder, pool_size=(layer[4], layer[4]), strides=(layer[5], layer[5]), 
                        padding=('SAME' if layer[6]==1 else 'VALID'), name = 'pooling_%d'%(layer_count))
                else:
                    op = tf.layers.max_pooling2d(op, pool_size=(layer[4], layer[4]), strides=(layer[5], layer[5]), 
                        padding=('SAME' if layer[6]==1 else 'VALID'), name = 'pooling_%d'%(layer_count))
                
                total_pool_sizes = total_pool_sizes + layer[4]**2
                total_pool_strides = total_pool_strides + layer[5]**2
                total_pool_paddings = total_pool_paddings + layer[6]

            elif layer[0] == 'fc':
                if first_layer == 1:
                    first_layer = 0
                    op = tf.layers.dense(inputs=tf_placeholder, units=layer[4], kernel_initializer=tf.ones_initializer(), 
                        activation=eval(activation_list[layer[5]]), use_bias=layer[6], name = 'dense_%d'%(layer_count))
                else:
                    op = tf.layers.dense(inputs=op, units=layer[4], kernel_initializer=tf.ones_initializer(), 
                        activation=eval(activation_list[layer[5]]), use_bias=layer[6], name = 'dense_%d'%(layer_count))
                
                total_fc_units = total_fc_units + layer[4]
                total_fc_acts = total_fc_acts + layer[5]
                total_fc_bias = total_fc_bias + layer[6]

            layer_count = layer_count + 1
        sess = tf.Session()
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)
        # Warm-up run
        for _ in range(20):#args.iter_warmup):
            sess.run(op, feed_dict={tf_placeholder: input_})
        # Benchmark run
        for _ in range(10):#args.iter_benchmark):
            start_time = time.time()
            sess.run(op, feed_dict={tf_placeholder: input_})
            time_list.append(((time.time()-start_time) * 1000))

        np_array_parameters = np.array(time_list)
        time_max = numpy.amax(np_array_parameters)
        time_min = numpy.amin(np_array_parameters)
        time_median = numpy.median(np_array_parameters)
        time_mean = numpy.mean(np_array_parameters)
        time_trim_mean = stats.trim_mean(np_array_parameters, 0.1)
        
        result_dict = {
            'batchsize' : batchsize,
            'input_image_size' : input_image_size,
            'input_image_channels' : input_image_channels,

            'total_conv_filters' : total_conv_filters,
            'total_conv_kernelsizes' : total_conv_kernelsizes,
            'total_conv_strides' : total_conv_strides,
            'total_conv_paddings' : total_conv_paddings,
            'total_conv_acts' : total_conv_acts,
            'total_conv_bias' : total_conv_bias,

            'total_pool_sizes' : total_pool_sizes,
            'total_pool_strides' : total_pool_strides,
            'total_pool_paddings' : total_pool_paddings,

            'total_fc_units' : total_fc_units,
            'total_fc_acts' : total_fc_acts,
            'total_fc_bias' : total_fc_bias,

            'time_max' : time_max,
            'time_min' : time_min,
            'time_median' : time_median,
            'time_mean' : time_mean,
            'time_trim_mean' : time_trim_mean,
        }
        print(result_dict)



