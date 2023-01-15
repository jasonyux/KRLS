import numpy as np
import random
import torch
import json
import argparse
import logging
import os

from typing import Union
from types import SimpleNamespace


class RecursiveNamespace(SimpleNamespace):
	"""aim: map nested dict to object
	https://dev.to/taqkarim/extending-simplenamespace-for-nested-dictionaries-58e8
	"""
	@staticmethod
	def map_entry(entry):
		if isinstance(entry, dict):
			return RecursiveNamespace(**entry)
		return entry
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		for key, val in kwargs.items():
			if type(val) == dict:
				setattr(self, key, RecursiveNamespace(**val))
			elif type(val) == list:
				setattr(self, key, list(map(self.map_entry, val)))
		return


def seed_everything(seed, deterministic=True):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	
	# works in latest pytorch==1.13.0 with gather being deterministic
	if deterministic:
		os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
		os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
		torch.use_deterministic_algorithms(True)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	return


def set_logger_file_handler_if_none(logger, file_path):
	if len(logger.handlers) == 1:
		file_handler = logging.FileHandler(file_path)
		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(logging.Formatter(
			fmt="%(asctime)s  [%(levelname)s] %(module)s; %(message)s",
			datefmt="%m/%d/%Y %H:%M:%S"))
		logger.addHandler(file_handler)
	return logger


def get_or_create_logger(logger_name=None, log_dir=None):
	logger = logging.getLogger(logger_name)

	# check whether handler exists
	if len(logger.handlers) > 0:
		return logger

	# set default logging level
	logger.setLevel(logging.DEBUG)

	# define formatters
	stream_formatter = logging.Formatter(
		fmt="%(asctime)s  [%(levelname)s] %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S")

	file_formatter = logging.Formatter(
		fmt="%(asctime)s  [%(levelname)s] %(module)s; %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S")

	# define and add handler
	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(logging.INFO)
	stream_handler.setFormatter(stream_formatter)
	logger.addHandler(stream_handler)

	if log_dir is not None:
		set_logger_file_handler_if_none(logger, os.path.join(log_dir, "log.txt"))
	return logger


def save_json(obj, save_path, indent=4):
	with open(save_path, "w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=indent)
	return


def cfg_to_dict(cfg):
	to_dict = {}
	for k, v in vars(cfg).items():
		if k.startswith('__'):
			continue
		to_dict[k] = v
	return to_dict


def dict_to_object(dict_data:dict):
	return RecursiveNamespace(**dict_data)


def save_cfg(cfg, save_path):
	if isinstance(cfg, dict):
		return save_json(cfg, save_path)
	# is a class
	to_dict = cfg_to_dict(cfg)
	return save_json(to_dict, save_path)


def str2bool(v) -> bool:
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def str2number(v) -> Union[int, float]:
	if isinstance(v, (int, float)):
		return v
	if v.isdigit():
		return int(v)
	try:
		return float(v)
	except:
		raise argparse.ArgumentTypeError('int of float expected.')


def str2none(v):
	if v.lower() == "none" or v.strip() == "":
		return None
	return v


def get_mean_std_with_padding(tensor:np.ndarray, padding_value=0):
	# get mean and std of a tensor, ignore padding
	tensor = tensor.reshape(-1)
	mask = tensor != padding_value
	mean = tensor[mask].mean()
	std = tensor[mask].std()
	return mean, std