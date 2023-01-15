from pathlib import Path

import numpy as np
import os

class MyDB:
	def __init__(self, db_base_dir, curr_batch_size):
		self.db_base_dir = db_base_dir
		self.tables = {
			'kl_logits': os.path.join(db_base_dir, 'kl_logits.npy'),
			'token_prob_logits': os.path.join(db_base_dir, 'token_prob_logits.npy')
		}
		self.curr_batch_size = curr_batch_size
		self.db_batch_size = -1
		if self.check_if_db_exists():
			with open(self.tables['kl_logits'], 'rb') as f:
				tmp = np.load(f)
				self.db_batch_size = tmp.shape[0]
			assert(self.curr_batch_size % self.db_batch_size == 0)
		return

	def check_if_db_exists(self, create=False):
		table_paths = list(self.tables.values())
		out = True
		for path in table_paths:
			if not os.path.exists(path):
				out = False
				# create
				if create:
					Path(path).touch()
					print(f"Created {path}")
					self.db_batch_size = self.curr_batch_size
		return out

	def search(self, table_name, row_idx):
		# adjust for batch size
		row_idx = row_idx * (self.curr_batch_size // self.db_batch_size)
		# the kl logit will be the same as the token prob logit
		with open(self.tables['kl_logits'], 'rb') as f:
			for _ in range(row_idx):
				np.load(f)
			out = []
			for _ in range(self.curr_batch_size // self.db_batch_size):
				out.append(np.load(f))
		out = np.concatenate(out, axis=0)
		return out