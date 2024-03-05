#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Friday, November 10th 2023, 12:25:19 pm
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

import logging
from logger_config import get_logger


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = get_logger(__name__)
# logging.basicConfig(
    # format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    # datefmt = "%m/%d/%Y %H:%M:%S",
    # level   = logging.INFO,
# )
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 


class Model(object):

    def __init__(self, rag_model, max_new_tokens=128):
        
        self.model     = rag_model
        self.max_new_tokens = max_new_tokens



    def generate(self, input_text):

        return self.model.generate(input_text, max_new_tokens=self.max_new_tokens)





