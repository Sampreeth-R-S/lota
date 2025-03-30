import json
import random

def select_random_entries(input_file, output_file, fraction=0.2):
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Ensure data is a list
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of entries.")
    
    # Select 20% of the entries randomly
    sampled_data = random.sample(data, int(len(data) * fraction))
    
    # Write the selected entries to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(sampled_data, f, indent=4)

# Example usage
select_random_entries('alpaca_eval_temp0.7.json', 'output.json')
