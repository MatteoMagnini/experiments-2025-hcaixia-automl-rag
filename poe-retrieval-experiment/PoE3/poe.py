import argparse
from pprint import pprint

from main import RunFramework
def main():
    parser = argparse.ArgumentParser(description="Out-of-topic Experiment.")

    # Define command-line arguments for each key in args_dict
    parser.add_argument('--name', type=str, default='out-of-topic-chatFAQ', help='Experiment name')
    parser.add_argument('--task', type=str, default='Classify a sentence as out of topic or not', help='Task description')
    parser.add_argument('--context', type=str, default=("""
        A supportive chatbot designed for pregnant women and new mothers, offering guidance from pregnancy through 
        the baby’s first 1,000 days. It provides answers on health, baby care, nutrition, psychological support 
        and financial planning, as well as assistance with related questions like choosing the best hospital for delivery.'"""
    ), help='Context for the task ')
    parser.add_argument('--expert_creator', type=str, default='Human Resource manager', help='Role of the expert creator (default=Human Resource manager)')
    parser.add_argument('--definition_framework', type=str, default="User Design Persona", help='Framework for expert description (default=User Design Persona)')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Model name to use for the experiment (default=meta-llama/Llama-3.2-1B-Instruct)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output files')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--temperature', type=float, default=1.2, help='Temperature setting for model generation (default=1.2)')
    parser.add_argument('--nucleus', type=float, default=0.9, help='Nucleus sampling parameter (default=0.9)')
    parser.add_argument('--alternatives', type=int, default=1, help='Number of alternative answers (default=1)')
    parser.add_argument('--resume', type=int, default=1, help='Resume option for experiment (default=1)')
    parser.add_argument('--cache_dir', type=str, default="/PoE3/.cache", help='Directory for cache files (default="/PoE3/.cache")')
    parser.add_argument('--max_experts_number', type=int, default=5, help='Maximum number of experts to use (default=5)')

    parser.add_argument('--baseline', type=bool, default=False, help='Exclude PoE3. Create Baseline using the bare model without characterize it')
    # Parse command-line arguments
    args = parser.parse_args()

    # Create args_dict from parsed arguments
    args_dict = vars(args)

    pprint(args_dict)
    print('\n\n\n\t\t\t\t***** Starting experiment ******\n\n\n')
    RunFramework(args_dict)

if __name__ == '__main__':
    main()
