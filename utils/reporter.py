import wandb
import copy


from utils.utils import get_or_create_logger, set_logger_file_handler_if_none, cfg_to_dict


logger = get_or_create_logger(__name__)

class SimpleReporter(object):
	def __init__(self, config):
		self.config = config
		self.global_step = 0
		self.log_frequency = 100
		self.states = {}
		# setup file logging if needed
		if self.config.log_file is not None:
			set_logger_file_handler_if_none(logger, self.config.log_file)
		return

	def dict_flatten_to_string(self, d):
		return ', '.join([f'{k}: {self.round_if_float(v)}' for k, v in d.items()])

	def update(self, data_dict):
		for k, v in data_dict.items():
			if isinstance(v, dict):
				for kk, vv in v.items():
					self.states[k][kk] += vv
			else:
				self.states[k] += v
		return

	def round_if_float(self, x):
		if isinstance(x, float):
			return f'{x:.3g}'
		else:
			return x

	def log(self, **non_volatile_data):
		output = ""
		for k, v in non_volatile_data.items():
			output += f'{k}: {self.round_if_float(v)} '
		
		for k, v in self.states.items():
			if isinstance(v, dict):
				output += f'[{k}]: {self.dict_flatten_to_string(v)} '
			else:
				output += f'[{k}]: {self.round_if_float(v)} '
		logger.info(output.strip())
		return

	def reset(self):
		self.states = {}
		return

	def step(self, data_dict, **non_volatile_data):
		self.global_step += 1
		# init
		if len(self.states) == 0:
			self.states = copy.deepcopy(data_dict)
		else:
			self.update(data_dict)

		if self.global_step % self.log_frequency == 0:
			self.log(**non_volatile_data)
			self.reset()
		return

	def flush(self, **non_volatile_data):
		self.log(**non_volatile_data)
		self.reset()
		return


class WandbReporter(SimpleReporter):
	def __init__(self, config, group_name, run_name=None, run=None):
		super().__init__(config)
		if run is None:
			self.run = self.__init_run(group_name, run_name)
		else:
			self.run = run
		self.run.define_metric(f"Validation.{config.val_watch_key}", summary='max')

	def __init_run(self, group_name, run_name):
		# object to dict
		config = cfg_to_dict(self.config)
		run = wandb.init(project='6998_convai', 
													group=group_name, 
													name=run_name, 
													config=config)
		return run

	def log(self, mode='train', **non_volatile_data):
		"""Actually flush the states
		"""
		loggable = {}
		for k, v in non_volatile_data.items():
			if isinstance(v, str):
				continue
			# self.run.log({mode: {k: v}}, commit=False)
			loggable[k] = v
		# log the rest to wandb
		loggable = {**loggable, **self.states}
		self.run.log({mode: loggable}, commit=True)
		# log to console
		super().log(**non_volatile_data)
		return

	def flush(self, mode='train', **non_volatile_data):
		self.log(mode, **non_volatile_data)
		self.reset()
		return