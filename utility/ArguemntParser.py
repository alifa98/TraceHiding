import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Script to train a model and perform unlearning.")

    parser.add_argument(
        '--model', 
        choices=['GRU', 'LSTM'], 
        help="Choose the model type: GRU or LSTM. Default is GRU.",
        nargs='?',  # Makes this argument optional
        default='LSTM'  # Default value if not provided
    )

    parser.add_argument(
        '--dataset', 
        type=str, 
        help="Specify the name of the dataset to be used. Default is 'HO_Geolife_Res8'.",
        nargs='?',  # Makes this argument optional
        default='Ho_Foursquare_NYC'
    )
    
    parser.add_argument(
        '--scenario',
        choices=['random', 'user'],
        help="Specify the scenario for unlearning. Default is 'user'.",
        nargs='?',  # Makes this argument optional
        default='user'
    )

    parser.add_argument(
        '--sampleSize', 
        type=int, 
        help="Specify the number of samples to be used for unlearning. Default is 50.",
        nargs='?',  # Makes this argument optional
        default=20
    )
    
    parser.add_argument(
        '--batchSize',
        type=int,
        help="Specify the batch size for training. Default is 25.",
        nargs='?',  # Makes this argument optional
        default=25
    )
    
    parser.add_argument(
        '--epochs', 
        type=int,
        help="Specify the number of epochs for the process. Default is 15.",
        nargs='?',  # Makes this argument optional
        default=15
    )
    
    parser.add_argument(
        '--plus', 
        type=bool,
        help="Specify whether to add reaminig data to gradient calculation (NegGrad+). Default is 'False'.",
        nargs='?',  # Makes this argument optional
        default=False
    )
    
    parser.add_argument(
        '--importance', 
        choices=['entropy', 'coverage_diversity'],
        type=str,
        help="Specify whether to add reaminig data to gradient calculation (NegGrad+). Default is 'False'.",
        nargs='?',  # Makes this argument optional
        default='entropy'
    )
    
    # For evaluation script
    parser.add_argument(
        '--method',
        choices=["original", "retraining", "trace_hiding", "finetune", "neg_grad", "neg_grad_plus", "bad-t", "scrub"],
        type=str,
        help="Specify the method for evaluation. Default is 'original'.",
        nargs='?',  # Makes this argument optional
        default='original'
    )
    
    parser.add_argument(
        '--epochIndex', 
        type=int,
        help="Specify the epoch number for evaluation. Default is 14.",
        nargs='?',  # Makes this argument optional
        default=14
    )

    args = parser.parse_args()
    
    return args
