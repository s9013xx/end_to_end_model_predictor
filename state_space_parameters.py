
# Posible Hyper-Parameters

#input
possible_batch_size = [1, 2, 4, 8, 16, 32, 64, 128, 256]
min_input_size = 32
max_input_size = 512
possible_input_channel = [1, 2, 3]
#convolutional
possible_conv_depth = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]    # 11     # Choices for number of filters in a convolutional layer
possible_conv_kernel_size = [1, 2, 3, 4, 5, 6, 7]                             # 7      # Choices for kernel size (square)
possible_conv_stride = [1, 2, 3, 4]                                    # 4 
possible_conv_padding = [0, 1]                                          # 2      # 0:VALID, 1:SAME
possible_conv_activate_function = [0, 1]                                # 2
possible_conv_bias = [0, 1]                                             # 2
#pooling
possible_pool_size = [1, 2, 3, 4, 5, 6, 7]                             # 7      # Choices for [kernel size, stride] for a max pooling layer
possible_pool_stride = [1, 2, 3, 4]                                    # 4
possible_pool_padding = [0, 1]                                          # 2      # 0:VALID, 1:SAME
#fully-connected
possible_fc_size = [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]      # 14 # Possible number of neurons in a fully connected layer
possible_fc_activate_function = [0, 1]                                  # 2
possible_fc_bias = [0, 1]                                               # 2