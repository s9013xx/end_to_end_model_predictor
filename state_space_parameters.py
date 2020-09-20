
# Transition Options
possible_conv_depths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]    # 11     # Choices for number of filters in a convolutional layer
possible_conv_sizes = [1, 2, 3, 4, 5, 6, 7]                             # 7      # Choices for kernel size (square)
possible_conv_strides = [1, 2, 3, 4]                                    # 4 
possible_conv_padding = [0, 1]                                          # 2      # 0:VALID, 1:SAME
possible_conv_activate_function = [0, 1]                                # 2
possible_conv_bias = [0, 1]                                             # 2

possible_pool_sizes = [1, 2, 3, 4, 5, 6, 7]                             # 7      # Choices for [kernel size, stride] for a max pooling layer
possible_pool_strides = [1, 2, 3, 4]                                    # 4
possible_pool_padding = [0, 1]                                          # 2      # 0:VALID, 1:SAME

possible_fc_sizes = [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]      # 14 # Possible number of neurons in a fully connected layer
possible_fc_activate_function = [0, 1]                                  # 2
possible_fc_bias = [0, 1]                                               # 2