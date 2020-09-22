import math
import numpy as np
import random
from operator import itemgetter
# import cnn

class State:
    def __init__(self,
                 layer_type=None,        # String -- start, conv, pool, fc
                 layer_depth=None,       # Current depth of network
                 batch_size=None,
                 input_image_size=None,  # Used for any layer that maintains square input (conv and pool), 0 otherwise
                 input_channel=None,
                 filter_depth=None,      # Used for conv, 0 when not conv
                 filter_size=None,       # Used for conv and pool, 0 otherwise
                 stride=None,            # Used for conv and pool, 0 otherwise
                 fc_size=None,           # Used for fc -- number of neurons in layer
                 terminate=None,
                 padding=None,
                 act=None,
                 bias=None,
                 output_image_size=None,
                 output_channel=None):

        self.layer_type = layer_type
        self.layer_depth = layer_depth
        self.batch_size = batch_size
        self.input_image_size = input_image_size
        self.input_channel = input_channel
        self.filter_depth = filter_depth
        self.filter_size = filter_size
        self.stride = stride
        self.fc_size = fc_size
        self.terminate = terminate
        self.padding=padding
        self.act=act
        self.bias=bias
        self.output_image_size=output_image_size
        self.output_channel=output_channel

    def as_tuple(self):
        return (self.layer_type,
                self.layer_depth,
                self.batch_size,
                self.input_image_size,
                self.input_channel,
                self.filter_depth,
                self.filter_size,
                self.stride,
                self.fc_size,
                self.terminate,
                self.padding,
                self.act,
                self.bias,
                self.output_image_size,
                self.output_channel)

    def as_list(self):
        return list(self.as_tuple())

    def copy(self):
        return State(self.layer_type,
                     self.layer_depth,
                     self.batch_size,
                     self.input_image_size,
                     self.input_channel,
                     self.filter_depth,
                     self.filter_size,
                     self.stride,
                     self.fc_size,
                     self.terminate,
                     self.padding,
                     self.act,
                     self.bias,
                     self.output_image_size,
                     self.output_channel)


class StateEnumerator:
    '''Class that deals with:
            Enumerating States (defining their possible transitions)
    '''
    def __init__(self, state_space_parameters, min_layer, max_layer):
        # Limits
        self.ssp = state_space_parameters
        self.min_layer = min_layer
        self.max_layer = max_layer

    def sample_conv_state(self, state):
        depth = self.ssp.possible_conv_depth[np.random.randint(len(self.ssp.possible_conv_depth))]
        kernel_size = self._possible_conv_sizes(state.input_image_size)[np.random.randint(len(self._possible_conv_sizes(state.input_image_size)))]
        stride = self.ssp.possible_conv_stride[np.random.randint(len(self.ssp.possible_conv_stride))]
        padding = self.ssp.possible_conv_padding[np.random.randint(len(self.ssp.possible_conv_padding))]
        act = self.ssp.possible_conv_activate_function[np.random.randint(len(self.ssp.possible_conv_activate_function))]
        bias = self.ssp.possible_conv_bias[np.random.randint(len(self.ssp.possible_conv_bias))]
        
        return State(layer_type='conv',
                        layer_depth=state.layer_depth + 1,
                        batch_size=state.batch_size,
                        input_image_size=state.output_image_size,
                        input_channel=state.output_channel,
                        filter_depth=depth,
                        filter_size=kernel_size,
                        stride=stride,
                        fc_size=0,
                        terminate=0,
                        padding=padding,
                        act=act,
                        bias=bias,
                        output_image_size=state.output_image_size if padding == 1 else self._calc_conv_new_image_size(state.output_image_size, kernel_size, stride),
                        output_channel=depth)


    def sample_pool_state(self, state):
        pool_size = self._possible_pool_sizes(state.input_image_size)[np.random.randint(len(self._possible_pool_sizes(state.input_image_size)))]
        stride = self.ssp.possible_pool_stride[np.random.randint(len(self.ssp.possible_pool_stride))]
        padding = self.ssp.possible_pool_padding[np.random.randint(len(self.ssp.possible_pool_padding))]

        return State(layer_type='pool',
                        layer_depth=state.layer_depth + 1,
                        batch_size=state.batch_size,
                        input_image_size=state.output_image_size,
                        input_channel=state.output_channel,
                        filter_depth=0,
                        filter_size=pool_size,
                        stride=stride,
                        fc_size=0,
                        terminate=0,
                        padding=padding,
                        act=0,
                        bias=0,
                        output_image_size=state.output_image_size if padding == 1 else self._calc_pool_new_image_size(state.output_image_size, pool_size, stride),
                        output_channel=state.output_channel)

    def sample_fc_state(self, state):
        fc_size = self.ssp.possible_fc_size[np.random.randint(len(self.ssp.possible_fc_size))]
        act = self.ssp.possible_fc_activate_function[np.random.randint(len(self.ssp.possible_fc_activate_function))]
        bias = self.ssp.possible_fc_bias[np.random.randint(len(self.ssp.possible_fc_bias))]

        if state.layer_type is not 'fc':
            input_dim = state.output_image_size**2
        else:
            input_dim = state.output_image_size

        return State(layer_type='fc',
                        layer_depth=state.layer_depth + 1,
                        batch_size=state.batch_size,
                        input_image_size=input_dim,
                        input_channel=state.output_channel,
                        filter_depth=0,
                        filter_size=0,
                        stride=0,
                        fc_size=fc_size,
                        terminate=0,
                        padding=0,
                        act=act,
                        bias=bias,
                        output_image_size=fc_size,
                        output_channel=1)

    def sample_terminate_state(self, state):
        return State(layer_type=state.layer_type,
                        layer_depth=state.layer_depth + 1,
                        batch_size=state.batch_size,
                        input_image_size=state.output_image_size,
                        input_channel=state.output_channel,
                        filter_depth=state.filter_depth,
                        filter_size=state.filter_size,
                        stride=state.stride,
                        fc_size=state.fc_size,
                        terminate=1,
                        padding=state.padding,
                        act=state.act,
                        bias=state.bias,
                        output_image_size=state.output_image_size,
                        output_channel=state.output_channel)

    def enumerate_state(self, state):
        '''
        Defines all legal state transitions:
           conv         -> conv, pool, fc               (IF state.layer_depth < layer_limit)
           pool         -> conv, pool, fc               (IF state.layer_depth < layer_limit)
           fc           -> fc                           (If state.layer_depth < layer_limit)
        '''
        if state.layer_depth == self.max_layer:
            return self.sample_terminate_state(state)

        if state.layer_type == 'fc':
            if state.layer_depth >= self.min_layer:
                sampled_layer_type = random.randint(0, 1)
            else:
                sampled_layer_type = 0

            if sampled_layer_type == 0:
                return self.sample_fc_state(state)
            else:
                return self.sample_terminate_state(state)
        else:
            if state.input_image_size <= 1:
                current_state = self.sample_fc_state(state)
            else:
                if state.layer_depth >= self.min_layer:
                    sampled_layer_type = random.randint(0, 3)
                else:
                    sampled_layer_type = random.randint(0, 2)
                # conv
                if sampled_layer_type == 0:
                    current_state = self.sample_conv_state(state)
                # pool
                if sampled_layer_type == 1:
                    current_state = self.sample_pool_state(state)
                # fc
                if sampled_layer_type == 2:
                    current_state = self.sample_fc_state(state)
                # terminate
                if sampled_layer_type == 3:
                    current_state = self.sample_terminate_state(state)

            return current_state

    def _calc_conv_new_image_size(self, image_size, filter_size, stride):
        '''Returns new image size given previous image size and filter parameters'''
        # new_size = int(math.ceil(float(image_size - filter_size + 1) / float(stride)))
        new_size = int(math.floor((image_size-filter_size)/stride)+1)
        return new_size

    def _possible_conv_sizes(self, image_size):
        possible_conv_kernel_size = [conv for conv in self.ssp.possible_conv_kernel_size if conv < image_size]
        # print("image_size:", image_size, "possible_conv_kernel_size:", possible_conv_kernel_size)
        return possible_conv_kernel_size

    def _possible_conv_strides(self, filter_size):
        return [stride for stride in self.ssp.possible_conv_stride if stride <= filter_size]

    def _calc_pool_new_image_size(self, image_size, filter_size, stride):
        '''Returns new image size given previous image size and filter parameters'''
        new_size = int(math.floor((image_size-filter_size)/stride)+1)
        return new_size

    def _possible_pool_sizes(self, image_size):
        possible_pool_size = [pool for pool in self.ssp.possible_pool_size if pool < image_size]
        # print("image_size:", image_size, "possible_pool_size:", possible_pool_size)
        return possible_pool_size

    def _possible_pool_strides(self, filter_size):
        return [stride for stride in self.ssp.possible_pool_stride if stride <= filter_size]

    def _possible_fc_size(self, state):
        '''Return a list of possible FC sizes given the current state'''
        if state.layer_type=='fc':
            return [i for i in self.ssp.possible_fc_size if i <= state.fc_size]
        return self.ssp.possible_fc_size

