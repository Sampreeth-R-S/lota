import json
from alpaca_eval import evaluate

# Load your JSON file
with open("/home/pabitracs/Sampreeth/lota/rlaif/scripts/alpaca_eval_temp0.0.json", "r") as f:
    data = json.load(f)

# Extract the instruction-output pairs
instances = [{"instruction": item["instruction"], "output": item["output"]} for item in data]

# Define model name
model_name = "Meta-Llama-3-8B"  # Change as needed
results = evaluate(
    model_name=model_name,
    responses=instances,
    reference_system="gpt-4-turbo"  # Default reference system
)

# Print the results
print(results)
