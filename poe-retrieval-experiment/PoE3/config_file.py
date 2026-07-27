import argparse
import json
from main import RunFramework

parser = argparse.ArgumentParser(description='Pool of Experts - command line interface.')
parser.add_argument('--config-file',
                    type=str,
                    help='Path to config file',
                    required=True)

# Parse the arguments
args = parser.parse_args()
config_file = args.config_file
#  load the config file( that is a json)
with open(config_file, 'r') as json_file:
    args_dict = json.load(json_file)

print('\n\n\n\t\t\t\t***** starting experiment ******\n\n\n')
RunFramework(args_dict)


#srun --partition=l40s --nodelist=fermi --gpus=1 --cpus-per-gpu=16 --mem=48GB --time=2:00:00 --qos=normal --container-image="/storage/IDA/Patrizio/hf_container.1.sqsh" --container-remap-root --container-mounts="/storage/IDA/Patrizio/:/Patrizio,/mnt/md0/data:/cache,/storage/IDA/Patrizio/ContextAugmentation/augment_context/xml_source_documents:/dataset" --pty bash
