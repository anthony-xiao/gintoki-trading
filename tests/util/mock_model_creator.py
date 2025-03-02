import h5py
import json
import numpy as np

def create_valid_mock_model():
    """Creates valid Keras-compatible HDF5 model"""
    with h5py.File('src/py/ml_core/models/regime_model.h5', 'w') as f:
        # Essential metadata
        f.attrs['backend'] = 'tensorflow'
        f.attrs['keras_version'] = '2.15.0'
        
        # Full model configuration
        model_config = {
            "class_name": "Sequential",
            "config": {
                "name": "sequential",
                "layers": [
                    {
                        "class_name": "InputLayer",
                        "config": {
                            "batch_input_shape": [None, 60, 20],
                            "dtype": "float32",
                            "sparse": False,
                            "name": "input_1"
                        }
                    },
                    {
                        "class_name": "LSTM",
                        "config": {
                            "name": "lstm",
                            "units": 64,
                            "activation": "tanh",
                            "recurrent_activation": "sigmoid",
                            "return_sequences": False
                        }
                    },
                    {
                        "class_name": "Dense",
                        "config": {
                            "name": "dense",
                            "units": 3,
                            "activation": "softmax"
                        }
                    }
                ]
            }
        }
        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
        
        # Weights structure
        weights = f.create_group("model_weights")
        
        # Input layer weights (empty)
        input_weights = weights.create_group("input_1")
        
        # LSTM weights
        lstm_weights = weights.create_group("lstm")
        lstm_weights.create_dataset("kernel:0", data=np.random.randn(20, 256))
        lstm_weights.create_dataset("recurrent_kernel:0", data=np.random.randn(64, 256))
        lstm_weights.create_dataset("bias:0", data=np.random.randn(256))
        
        # Dense weights
        dense_weights = weights.create_group("dense")
        dense_weights.create_dataset("kernel:0", data=np.random.randn(64, 3))
        dense_weights.create_dataset("bias:0", data=np.random.randn(3))

if __name__ == "__main__":
    create_valid_mock_model()
