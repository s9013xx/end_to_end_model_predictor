# End to End Model Predictor

This project is used to prediction the inference latency of end-to-end neural network.

Step 1. Generate the end-to-end model with different layers : 

    python3 model_generator.py --min_layer 1 --max_layer 5

Step 2. Execute the end-to-end model: 

    python3 model_executor.py

Step 3. Transform the features for model training : 

    python3 feature_transformer.py

Step 4. Train the end-to-end model inference latency predictor : 

    python3 model_trainer.py


