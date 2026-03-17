
import os
import json
from datetime import datetime

def log_parameters(params, log_file_path='param_log.json'):
    """
    Logs program parameters to a specified log file in JSON format.

    This function checks if the specified log file exists. If not, it initializes 
    the log file with an empty list. It then appends the provided parameters as 
    a dictionary to the log file, along with a timestamp indicating when the 
    parameters were logged.

    Args:
        params (dict): A dictionary containing the parameters to be logged. 
                       Each key-value pair in the dictionary represents a parameter 
                       name and its corresponding value.
        log_file_path (str): The file path where the log should be stored. 
                             Default is 'param_log.json'.

    # Example usage
    log_parameters(
        params={'batch_size': 32, 'num_epochs': 220, 'model_name': 'example_model'},
        log_file_path='param_log.json',
    )
    """
    # Check if the log file exists, else make it
    if not os.path.exists(log_file_path):
        # If the file does not exist, initialize it with an empty list
        with open(log_file_path, 'w') as log_file:
            json.dump([], log_file, indent=4)

    # Load the existing log
    with open(log_file_path, 'r') as log_file:
        log_data = json.load(log_file)

    # Append the new parameters with a timestamp
    entry = {
        "timestamp": datetime.now().isoformat(),
        "parameters": params
    }
    log_data.insert(0, entry)

    # Write the updated log back to the file
    with open(log_file_path, 'w') as log_file:
        json.dump(log_data, log_file, indent=4)


