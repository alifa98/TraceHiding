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
        default='HO_Geolife_Res8'
    )

    parser.add_argument(
        '--sampleSize', 
        type=int, 
        help="Specify the number of samples to be used for unlearning. Default is 50.",
        nargs='?',  # Makes this argument optional
        default=50
    )
    
    parser.add_argument(
        '--batchSize',
        type=int,
        help="Specify the batch size for training. Default is 25.",
        nargs='?',  # Makes this argument optional
        default=25
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
        type=str,
        help="Specify whether to add reaminig data to gradient calculation (NegGrad+). Default is 'False'.",
        nargs='?',  # Makes this argument optional
        default='entropy'
    )

    args = parser.parse_args()
    
    return args
