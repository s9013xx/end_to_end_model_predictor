# End to End Model Predictor

This project is used to prediction the inference latency of end-to-end neural network.

Step 1. Generate the end-to-end model with different layers : 

    python3 model_generator.py --min_layer 1 --max_layer 7 -net_num 10000

Step 2. Execute the end-to-end model: 

    python3 network_executor.py --min_layer 1 --max_layer 7 -net_num 10000

Step 3. Transform the features to one-hot for model training : 

    python network_extractor.py --min_layer 1 --max_layer 7 -net_num 10000

    python network_features_to_one_hot.py --min_layer 1 --max_layer 7 -net_num 10000

Step 4. Train the end-to-end model inference latency predictor : 

    python3 model_trainer.py


