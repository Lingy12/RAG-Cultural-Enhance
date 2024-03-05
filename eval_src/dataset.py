#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Monday, July 24th 2023, 11:58:08 am
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

import os

# add parent directory to sys.path
import sys
sys.path.append('.')

import json

import random
import logging

from datasets import load_dataset

from dataset_src.sg_eval import sg_eval_dataset
from dataset_src.us_eval import us_eval_dataset
from dataset_src.ph_eval import ph_eval_dataset
from logger_config import get_logger



# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = get_logger(__name__)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 


class Dataset(object):
    
    def __init__(self, dataset_name: str="", prompt_index: int=1, support_langs: list=[], eval_mode: str=None):

        self.prompt_index    = prompt_index
        self.dataset_name    = dataset_name
        self.support_langs   = support_langs
        self.eval_mode       = eval_mode

        self.load_dataset()
        self.data_format()


    def load_dataset(self):

        logger.info("Loading dataset: {}".format(self.dataset_name))

        
        if self.dataset_name in ['open_sg_qa', 'sing2eng', 'cross_xquad']:
            # Load local dataset
            full_path = os.path.join('data', self.dataset_name+'.json')
            with open(full_path, 'r', encoding="utf-8") as f:
                full_data = json.load(f)
        else:
            # Load from HuggingFace
            full_data = load_dataset('SeaEval/SeaEval_datasets', self.dataset_name, split='test')
            full_data = [sample for sample in full_data]


        self.raw_data = full_data

        logger.info("The dataset originally has {} samples".format(len(full_data)))
        logger.info("Loaded {} samples for evaluation".format(len(self.raw_data)))
     

    def data_format(self):
        if self.dataset_name == 'sg_eval':
            self.dataset_processor = sg_eval_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'us_eval':
            self.dataset_processor = us_eval_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()
        
        elif self.dataset_name == 'ph_eval':
            self.dataset_processor = ph_eval_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        else:
            raise NotImplementedError("Dataset {} not implemented yet".format(self.dataset_name))

