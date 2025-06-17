from argparse import ArgumentParser
from typing import Dict
from datetime import datetime

def return_parser() -> Dict:

    parser = ArgumentParser(description='Statkraft Assessment')

    parser.add_argument('--mode', type=str, default='analysis', help='Choose between analysis, forecast')
    parser.add_argument('--val_start', type=str, default='2014-01-01', help='Start date for validation data')
    parser.add_argument('--test_start', type=str, default='2018-01-01', help='Start date for test data')
    parser.add_argument('--test_end', type=str, default='2018-12-31', help='Start date for test data')


    args = parser.parse_args()
    args_dict =  vars(args)

    # Format valuedate
    args_dict['val_start'] = datetime.strptime(args_dict['val_start'], '%Y-%m-%d')
    args_dict['test_start'] = datetime.strptime(args_dict['test_start'], '%Y-%m-%d')
    args_dict['test_end'] = datetime.strptime(args_dict['test_end'], '%Y-%m-%d')

    return args_dict