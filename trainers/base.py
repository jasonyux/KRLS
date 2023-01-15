from abc import ABCMeta, abstractmethod


class BaseTrainer(metaclass=ABCMeta):
	def __init__(self, reader, config, model):
		self.cfg = config
		self.reader = reader
		self.model = model

	@abstractmethod
	def save_model(self, epoch):
		raise NotImplementedError

	@abstractmethod
	def train(self):
		raise NotImplementedError

	@abstractmethod
	def predict(self):
		raise NotImplementedError

class MultiWOZTrainer(BaseTrainer):
	def __init__(self, reader, config, model):
		super(MultiWOZTrainer, self).__init__(reader, config, model)

	def save_model(self, epoch):
		pass

	def train(self):
		pass

	def predict(self):
		pass