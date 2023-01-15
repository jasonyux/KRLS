import torch
from abc import abstractmethod

class ModelBase(torch.nn.Module):
	def __init__(self, config):
		super(ModelBase, self).__init__()
		self.cfg = config

	def forward(self, *args, **kwargs):
		raise NotImplementedError

	@abstractmethod
	def get_optimizer_and_scheduler(self, *args, **kwargs):
		raise NotImplementedError