import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import random
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from arguments import get_args
import init_path
from utils_bc import utils

from utils_bc import utils_interactive_eval
from utils_bc.utils import save_model, load_pretrained_model
from utils_bc.utils_llm import get_pretrained_tokenizer
from interactive_collection import data_collection


def get_logger(args, log_path):
    if os.path.exists(log_path):
        os.remove(log_path)

    import logging
    a_logger = logging.getLogger()
    a_logger.setLevel(logging.INFO)

    output_file_handler = logging.FileHandler(log_path)
    stdout_handler = logging.StreamHandler(sys.stdout)

    a_logger.addHandler(output_file_handler)
    a_logger.addHandler(stdout_handler)
    logging = a_logger
    return logging


def main():
    args = get_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    args = init_path.get_logger_path(args)
    logging = get_logger(args, args.log_path)
    
    ## initial path
    args = init_path.initialize_path(args)
    args = init_path.load_data_info(args)
    
    ## Testing
    vh_envs = utils_interactive_eval.connect_env(args, logging)
    interactive_eval_success_rate = data_collection(args, vh_envs, logging=logging)
    
if __name__ == "__main__":
    main()
