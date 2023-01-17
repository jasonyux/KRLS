import glob
import os
import shutil
import torch
import copy
import re
import json
import numpy as np

from pathlib import Path
from typing import *
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import BaseModelOutput
from tqdm.auto import tqdm
from collections import OrderedDict, defaultdict
from mwzeval.metrics import Evaluator

from trainers.base import BaseTrainer
from data_utils.mttod.reader import MultiWOZIterator
from data_utils.mttod.utils import definitions
from models.mttod import MTTODModel
from utils.reporter import WandbReporter, SimpleReporter
from utils.utils import seed_everything, save_json, get_or_create_logger


logger = get_or_create_logger(__name__)


class MTTODTrainer(BaseTrainer):
	def __init__(self, dataset, cfg, model:MTTODModel, test_dataset=None):
		super(MTTODTrainer, self).__init__(dataset, cfg, model)

		self.iterator = MultiWOZIterator(dataset, cfg.data_dir)
		if test_dataset is not None:
			self.test_iterator = MultiWOZIterator(test_dataset, cfg.data_dir)
		self.reporter = SimpleReporter(cfg)	# calls update() and log()
		self.val_reporter = SimpleReporter(cfg) # global_step%logging_freq should never trigger logging
		self.val_reporter.log_frequency = float("inf")
		seed_everything(cfg.seed, deterministic=cfg.deterministic)

		self.__special_token_ids = [v for k, v in dataset.tokenizer.get_added_vocab().items() if k.startswith('[') and not k.startswith('[db')]
		self._best_performances:DefaultDict = defaultdict(lambda: [])
		self.is_policy_optimization = False
		self.__evaluator = Evaluator(bleu=True, success=True, richness=True, dst=True)

		self._train_batches = None  # used for debugging

	def save_model(self, epoch, val_output):
		val_watch_key = self.cfg.val_watch_key.split('.')
		latest_ckpt = "ckpt-epoch{}".format(epoch)
		save_path = os.path.join(self.cfg.model_dir, latest_ckpt)
		'''
		if self.cfg.num_gpus > 1:
			model = self.model.module
		else:
			model = self.model
		'''
		model = self.model

		if val_output is None:
			model.save_pretrained(save_path)

			# keep chekpoint up to maximum
			checkpoints = sorted(
				glob.glob(os.path.join(self.cfg.model_dir, "ckpt-*")),
				key=os.path.getmtime,
				reverse=True)

			checkpoints_to_be_deleted = checkpoints[self.cfg.max_to_keep_ckpt:]

			for ckpt in checkpoints_to_be_deleted:
				shutil.rmtree(ckpt)
		else:
			v = val_output
			val_watch_key_str = '_'.join(val_watch_key)
			for k in val_watch_key:
				v = v.get(k)
				curr_performance = v
			if isinstance(curr_performance, torch.Tensor):
				curr_performance = curr_performance.item()
			
			if len(self._best_performances[val_watch_key_str]) < self.cfg.max_to_keep_ckpt or \
				curr_performance > self._best_performances[val_watch_key_str][-1]: # -1 is the worst performance among bests
				self._best_performances[val_watch_key_str].insert(0, curr_performance)
				self._best_performances['mapping'].insert(0, (curr_performance, latest_ckpt))
				model.save_pretrained(save_path)

			if len(self._best_performances[val_watch_key_str]) > self.cfg.max_to_keep_ckpt:
				logger.info(f"Deleting the last from {self._best_performances['mapping']}")
				_, ckpt_to_be_deleted = self._best_performances['mapping'].pop(-1)
				self._best_performances[val_watch_key_str].pop(-1)
				shutil.rmtree(os.path.join(self.cfg.model_dir, ckpt_to_be_deleted))
				logger.info(f"Deleted {ckpt_to_be_deleted}, now {self._best_performances['mapping']}")
			
			# sort best performances
			self._best_performances['mapping'] = sorted(self._best_performances['mapping'], key=lambda x: x[0], reverse=True)
			self._best_performances[val_watch_key_str] = [x[0] for x in self._best_performances['mapping']]
		return latest_ckpt

	def count_tokens(self, pred, label, pad_id):
		pred = pred.view(-1)
		label = label.view(-1)

		num_count = label.ne(pad_id).long().sum()
		num_correct = torch.eq(pred, label).long().sum()

		return num_correct, num_count

	def get_attention_mask_span(self, resp_label_ids, bos_token_id, eos_token_id, include_token_id=True):
		"""get the attention mask for the span of the response
		"""
		resp_start_id = torch.where(resp_label_ids == bos_token_id)[1]
		resp_end_id = torch.where(resp_label_ids == eos_token_id)[1]
		resp_attention_mask = torch.zeros_like(resp_label_ids)
		for i, (sid, eid) in enumerate(zip(resp_start_id, resp_end_id)):
			if include_token_id:
				# bos_resp and eos_resp are included
				resp_attention_mask[i, sid: eid+1] = 1
			else:
				# bos_resp and eos_resp are excluded
				resp_attention_mask[i, sid+1: eid] = 1
		return resp_attention_mask

	def count_special_tokens(self, pred, resp_label, pad_id):
		pred_np = pred.detach().cpu().numpy()
		resp_label_np = resp_label.detach().cpu().numpy()

		all_special_tokens = self.model.model._special_token_ids # + self.model.model._da_token_ids
		bos_resp_token_id = self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
		eos_resp_token_id = self.reader.get_token_id(definitions.EOS_RESP_TOKEN)

		special_token_mask = np.isin(resp_label_np, all_special_tokens)
		resp_token_mask = self.get_attention_mask_span(resp_label, bos_resp_token_id, eos_resp_token_id, include_token_id=False)
		resp_token_mask = resp_token_mask.detach().cpu().numpy()

		total_tokens = resp_token_mask.sum().item()
		correct_tokens = resp_token_mask * (pred_np == resp_label_np)
		total_correct_tokens = correct_tokens.sum().item()
		correct_token_perf = total_correct_tokens / total_tokens

		total_special_tokens = special_token_mask.sum().item()
		correct_special_tokens = special_token_mask * correct_tokens
		total_correct_special_tokens = correct_special_tokens.sum().item()
		special_token_perf = total_correct_special_tokens/total_special_tokens if total_special_tokens > 0 else 0

		total_non_special_tokens = total_tokens - total_special_tokens
		correct_non_special_tokens = (1 - special_token_mask) * correct_tokens
		total_correct_non_special_tokens = correct_non_special_tokens.sum().item()
		non_special_token_perf = total_correct_non_special_tokens/total_non_special_tokens if total_non_special_tokens > 0 else 0

		return correct_token_perf, special_token_perf, non_special_token_perf

	def get_tod_scale(self, resp_pred, resp_label):
		resp_pred = resp_pred.detach().cpu().numpy()
		resp_label = resp_label.detach().cpu().numpy()

		special_token_mask = np.isin(resp_label, self.__special_token_ids)
		num_critical_errors = np.sum(special_token_mask * (resp_pred != resp_label))
		# divide by batch size
		return num_critical_errors / resp_label.shape[0]

	def step_fn(self, inputs, span_labels, belief_labels, resp_labels):
		inputs = inputs.to(self.cfg.device)
		span_labels = span_labels.to(self.cfg.device)
		belief_labels = belief_labels.to(self.cfg.device)
		resp_labels = resp_labels.to(self.cfg.device)
		# for multitask learning, we essentially need to:
		# 1. forwrad pass the encoder
		# 2.1 do span prediction from encoder_hidden + span_head
		# 2.2 do belief prediction from encoder_hidden -> decdoder -> decoder_hidden -> lm_head
		# 2.3 do response prediction from encoder_hidden -> resp_decoder -> resp_decoder_hidden -> resp_lm_head
		attention_mask = torch.where(inputs == self.reader.pad_token_id, 0, 1)
		belief_outputs = self.model(input_ids=inputs,
							attention_mask=attention_mask,
							span_labels=span_labels,
							lm_labels=belief_labels,
							return_dict=False,
							add_auxiliary_task=True,
							decoder_type="belief")

		belief_loss = belief_outputs[0]
		belief_pred = belief_outputs[1].detach()

		span_loss = belief_outputs[2]
		span_pred = belief_outputs[3].detach()

		last_hidden_state = belief_outputs[5]
		
		encoder_outputs = BaseModelOutput(last_hidden_state=last_hidden_state)
		resp_outputs = self.model(attention_mask=attention_mask,
							encoder_outputs=encoder_outputs,
							lm_labels=resp_labels,
							return_dict=False,
							decoder_type="resp")

		resp_loss = resp_outputs[0]
		resp_pred = resp_outputs[1].detach()

		# loss = belief_loss + (self.cfg.aux_loss_coeff * span_loss) + (self.cfg.resp_loss_coeff * resp_loss)

		tod_loss_scale = 0.0
		if self.cfg.use_tod_loss_scale:
			tod_loss_scale = self.get_tod_scale(resp_pred, resp_labels)
		
		loss = belief_loss + (self.cfg.aux_loss_coeff * span_loss) + ((self.cfg.resp_loss_coeff + tod_loss_scale) * resp_loss)

		## checking acc
		num_belief_correct, num_belief_count = self.count_tokens(
			belief_pred, belief_labels, pad_id=self.reader.pad_token_id)
		num_span_correct, num_span_count = self.count_tokens(
				span_pred, span_labels, pad_id=self.reader.pad_token_id)
		num_resp_correct, num_resp_count = self.count_tokens(
				resp_pred, resp_labels, pad_id=self.reader.pad_token_id)
		correct_token_perf, special_token_perf, non_special_token_perf = self.count_special_tokens(
			resp_pred, resp_labels, pad_id=self.reader.pad_token_id
		)
		step_outputs = {
			"belief": {
				"loss": belief_loss.item(),
				"correct": num_belief_correct.item(),
				"count": num_belief_count.item(),
			},
			"span": {
				"loss": span_loss.item(),
				"correct": num_span_correct.item(),
				"count": num_span_count.item(),
			},
			"resp": {
				"loss": resp_loss.item(),
				"correct": num_resp_correct.item(),
				"count": num_resp_count.item(),
				"percent_special_correct": special_token_perf,
				"percent_normal_correct": non_special_token_perf,
				"percent_correct": correct_token_perf
			},
		}
		if self.cfg.use_tod_loss_scale:
			step_outputs["tod_loss_scale"] = tod_loss_scale / self.reporter.log_frequency,
		return loss, step_outputs

	def train_epoch(self, train_iterator, optimizer, scheduler, reporter=None):
		self.model.train()
		self.model.zero_grad()

		for step, batch in enumerate(train_iterator):
			inputs, labels = batch

			_, belief_labels, _ = labels

			loss, step_outputs = self.step_fn(inputs, *labels)

			if self.cfg.grad_accum_steps > 1:
				loss = loss / self.cfg.grad_accum_steps

			loss.backward()

			torch.nn.utils.clip_grad_norm_(
				self.model.parameters(), self.cfg.max_grad_norm)

			if (step + 1) % self.cfg.grad_accum_steps == 0:
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()

				lr = scheduler.get_last_lr()[0]

				if reporter is not None:
					reporter.step(step_outputs, lr=lr)
		return

	def train(self):
		train_batches, num_training_steps_per_epoch, _, _ = self.iterator.get_batches(
			"train", self.cfg.batch_size, self.cfg.num_gpus, shuffle=True,
			num_dialogs=self.cfg.num_train_dialogs, excluded_domains=None)

		optimizer, scheduler = self.model.get_optimizer_and_scheduler(num_training_steps_per_epoch)

		for epoch in range(1, self.cfg.epochs + 1):
			train_iterator = self.iterator.get_data_iterator(
				train_batches, 'e2e', False, True, -1, resp_use_bspn=self.cfg.resp_use_bspn)

			self.train_epoch(train_iterator, optimizer, scheduler, self.reporter)

			logger.info("done {}/{} epoch".format(epoch, self.cfg.epochs))

			val_output = None
			if not self.cfg.no_validation:
				val_output = self.validation(self.reporter.global_step, epoch)
			
			if self.cfg.save_model:
				self.save_model(epoch, val_output=val_output)
		return

	def get_score_per_dialog(self, predictions):
		for did, dialog in predictions.items():
			last_turn = len(dialog) - 1
			for t in range(len(dialog)):
				mini_dialog = {
					did: dialog[:t+1]
				}
				score = self.__evaluator.evaluate(copy.deepcopy(mini_dialog))
				total_score = score['bleu']['mwz22'] + (score['success']['inform']['total'] + score['success']['success']['total']) / 2.0
				score['total_score'] = total_score

				if t == last_turn:
					predictions[did][t]['score'] = score
				else:
					score.pop('dst')
					score.pop('richness')
					predictions[did][t]['score'] = score
		return predictions

	def validation(self, global_step, epoch) -> dict:
		self.model.eval()
		self.val_reporter.reset()

		if self._train_batches is not None:
			train_batches = self._train_batches
		else:
			train_batches, num_steps, _, _ = self.iterator.get_batches(
				"train", self.cfg.batch_size, self.cfg.num_gpus, shuffle=True, seed=self.cfg.seed
			)
		dev_batches, num_steps, _, _ = self.iterator.get_batches(
			"dev", self.cfg.batch_size, self.cfg.num_gpus, shuffle=False, seed=self.cfg.seed)
		
		if self.cfg.debug or self.cfg.pilot_run:
			multiplier = 2 if self.cfg.batch_size < 8 else 1
			cut_off = 5 if self.cfg.debug else 50
			cut_off *= multiplier
			dev_batches = dev_batches[:cut_off]
			
			num_steps = 0
			dev_iterator = self.iterator.get_data_iterator(
				dev_batches, self.cfg.task, self.cfg.ururu, 
				self.cfg.add_auxiliary_task, self.cfg.context_size, resp_use_bspn=self.cfg.resp_use_bspn
			)
			for step in dev_iterator:
				num_steps += 1

		dev_iterator = self.iterator.get_data_iterator(
			dev_batches, self.cfg.task, self.cfg.ururu, self.cfg.add_auxiliary_task, self.cfg.context_size, resp_use_bspn=self.cfg.resp_use_bspn)

		with torch.no_grad():
			for batch in tqdm(dev_iterator, total=num_steps, desc="Validation"):
				inputs, labels = batch

				_, step_outputs = self.step_fn(inputs, *labels)

				self.val_reporter.step(step_outputs)

			val_performance = copy.deepcopy(self.val_reporter.states)

			# check if training learned train_batches
			train_batches: list = train_batches[:10]
			train_results = self.predict_batches(train_batches, keep_raw=True)
			train_score = self.__evaluator.evaluate(copy.deepcopy(train_results))  # evaluator will modify results
			train_score['total_score'] = train_score['bleu']['mwz22'] + (train_score['success']['inform']['total'] + train_score['success']['success']['total']) / 2.0
			if self.cfg.score_each_dialog:
				train_results = self.get_score_per_dialog(train_results)
			
			# test multiwoz score
			test_results = self.predict_batches(dev_batches, keep_raw=True)
			test_score = self.__evaluator.evaluate(copy.deepcopy(test_results))  # evaluator will modify results
			test_score['total_score'] = test_score['bleu']['mwz22'] + (test_score['success']['inform']['total'] + test_score['success']['success']['total']) / 2.0

			if self.cfg.skip_val_predictions:
				results = copy.deepcopy(test_results)
				score = copy.deepcopy(test_score)
				score['total_score'] = test_score['total_score']
			else:
				results = self.predict_batches(dev_batches, keep_raw=True)
				score = self.__evaluator.evaluate(copy.deepcopy(results))  # evaluator will modify results
				score['total_score'] = score['bleu']['mwz22'] + (score['success']['inform']['total'] + score['success']['success']['total']) / 2.0
			
			score['train_scores'] = train_score
			score['test_scores'] = test_score
			if self.cfg.score_each_dialog:
				results = self.get_score_per_dialog(results)

		if self.cfg.save_val_predictions:
			# create necessary directories
			latest_ckpt = "val-epoch{}.json".format(epoch)
			save_dir = os.path.join(self.cfg.model_dir, "preds")
			Path(save_dir).mkdir(parents=True, exist_ok=True)

			save_path = os.path.join(save_dir, latest_ckpt)
			perf = {'predictions': results, 'score': score}
			save_json(perf, save_path)
			
			# save train as well
			latest_ckpt = "train-epoch{}.json".format(epoch)
			save_path = os.path.join(save_dir, latest_ckpt)
			perf = {'predictions': train_results, 'score': train_score}
			save_json(perf, save_path)
		
		self.val_reporter.log(mode="Validation", **score) # flush

		val_performance = {**val_performance, **score}
		return val_performance

	def finalize_bspn(self, belief_outputs, domain_history, constraint_history, span_outputs=None, input_ids=None):
		bos_token_id = self.reader.get_token_id(definitions.BOS_BELIEF_TOKEN)
		eos_token_id = self.reader.get_token_id(definitions.EOS_BELIEF_TOKEN)

		batch_decoded = []
		for i, belief_output in enumerate(belief_outputs):
			if belief_output[0] == self.reader.pad_token_id:
				belief_output = belief_output[1:]

			if eos_token_id not in belief_output:
				eos_idx = len(belief_output) - 1
			else:
				eos_idx = belief_output.index(eos_token_id)

			bspn = belief_output[:eos_idx + 1]
			# remove all the paddings
			bspn = [x for x in bspn if x != self.reader.pad_token_id]

			decoded = {}

			decoded["bspn_gen"] = bspn

			# update bspn using span output
			# try:
			if span_outputs is not None and input_ids is not None:
				span_output = span_outputs[i]
				input_id = input_ids[i]
				eos_idx = input_id.index(self.reader.eos_token_id)
				input_id = input_id[:eos_idx]

				# skip if span output is garbage
				if any([o not in self.reader.span_tokens for o in span_output]) \
					or len(span_output) > len(input_id):
					if bos_token_id not in bspn:
						bspn = [bos_token_id] + bspn
					if eos_token_id not in bspn:
						bspn = bspn + [eos_token_id]
					decoded["bspn_gen_with_span"] = bspn
					batch_decoded.append(decoded)
					continue

				span_result = {}

				bos_user_id = self.reader.get_token_id(definitions.BOS_USER_TOKEN)

				span_output = span_output[:eos_idx]

				b_slot = None
				for t, span_token_idx in enumerate(span_output):
					turn_id = max(input_id[:t].count(bos_user_id) - 1, 0)
					turn_domain = domain_history[i][turn_id]

					if turn_domain not in definitions.INFORMABLE_SLOTS:
						continue

					span_token = self.reader.span_tokens[span_token_idx]

					if span_token not in definitions.INFORMABLE_SLOTS[turn_domain]:
						b_slot = span_token
						continue

					if turn_domain not in span_result:
						span_result[turn_domain] = defaultdict(list)

					if b_slot != span_token:
						span_result[turn_domain][span_token] = [input_id[t]]
					else:
						span_result[turn_domain][span_token].append(input_id[t])

					b_slot = span_token

				for domain, sv_dict in span_result.items():
					for s, v_list in sv_dict.items():
						value = v_list[-1]
						span_result[domain][s] = self.reader.tokenizer.decode(
							value, clean_up_tokenization_spaces=False)

				span_dict = copy.deepcopy(span_result)

				ontology = self.reader.db.extractive_ontology

				flatten_span = []
				for domain, sv_dict in span_result.items():
					flatten_span.append("[" + domain + "]")

					for s, v in sv_dict.items():
						if domain in ontology and s in ontology[domain]:
							if v not in ontology[domain][s]:
								del span_dict[domain][s]
								continue

						if s == "destination" or s == "departure":
							_s = "destination" if s == "departure" else "departure"

							if _s in sv_dict and v == sv_dict[_s]:
								if s in span_dict[domain]:
									del span_dict[domain][s]
								if _s in span_dict[domain]:
									del span_dict[domain][_s]
								continue

						if s in ["time", "leave", "arrive"]:
							v = v.replace(".", ":")
							if re.match("[0-9]+:[0-9]+", v) is None:
								del span_dict[domain][s]
								continue
							else:
								span_dict[domain][s] = v

						flatten_span.append("[value_" + s + "]")
						flatten_span.append(v)

					if len(span_dict[domain]) == 0:
						del span_dict[domain]
						flatten_span.pop()

				decoded["span"] = flatten_span

				constraint_dict = self.reader.bspn_to_constraint_dict(
					self.reader.tokenizer.decode(bspn, clean_up_tokenization_spaces=False))

				_constraint_dict = copy.deepcopy(constraint_dict)

				bspn_gen_with_span = self.reader.constraint_dict_to_bspn(
					_constraint_dict)

				bspn_gen_with_span = self.reader.encode_text(
					bspn_gen_with_span,
					bos_token=definitions.BOS_BELIEF_TOKEN,
					eos_token=definitions.EOS_BELIEF_TOKEN)

				decoded["bspn_gen_with_span"] = bspn_gen_with_span

			batch_decoded.append(decoded)
			# except:
			# 	if bos_token_id not in bspn:
			# 		bspn = [bos_token_id] + bspn
			# 	if eos_token_id not in bspn:
			# 		bspn = bspn + [eos_token_id]
			# 	decoded["bspn_gen_with_span"] = bspn
			# 	batch_decoded.append(decoded)
		return batch_decoded

	def finalize_resp(self, resp_outputs, keep_raw=False):
		bos_action_token_id = self.reader.get_token_id(definitions.BOS_ACTION_TOKEN)
		eos_action_token_id = self.reader.get_token_id(definitions.EOS_ACTION_TOKEN)

		bos_resp_token_id = self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
		eos_resp_token_id = self.reader.get_token_id(definitions.EOS_RESP_TOKEN)

		batch_decoded = []
		for resp_output in resp_outputs:
			resp_output: list = resp_output[1:]
			if self.reader.eos_token_id in resp_output:
				eos_idx = resp_output.index(self.reader.eos_token_id)
				resp_output = resp_output[:eos_idx]
			if keep_raw:
				raw_resp = resp_output.copy()

			try:
				bos_action_idx = resp_output.index(bos_action_token_id)
				eos_action_idx = resp_output.index(eos_action_token_id)
			except ValueError:
				logger.debug("bos/eos action token not in : {}".format(self.reader.tokenizer.decode(resp_output)))
				aspn = [bos_action_token_id, eos_action_token_id]
			else:
				aspn = resp_output[bos_action_idx:eos_action_idx + 1]

			try:
				bos_resp_idx = resp_output.index(bos_resp_token_id)
				eos_resp_idx = resp_output.index(eos_resp_token_id)
			except ValueError:
				logger.debug("bos/eos resp token not in : {}".format(self.reader.tokenizer.decode(resp_output)))
				resp = [bos_resp_token_id, eos_resp_token_id]
			else:
				resp = resp_output[bos_resp_idx:eos_resp_idx + 1]

			decoded = {"aspn_gen": aspn, "resp_gen": resp}
			if keep_raw:
				decoded["resp_raw_gen"] = raw_resp

			batch_decoded.append(decoded)
		return batch_decoded

	def bspn_to_constraint_dict(self, bspn):
		bspn = bspn.replace('<bos_belief>', '')
		bspn = bspn.replace('<eos_belief>', '')
		bspn = bspn.strip().split()

		constraint_dict = {}
		domain, slot = None, None
		for token in bspn:
			if token.startswith('['):
				token = token[1:-1]

				if token.startswith('value_'):
					if domain is None:
						continue
					if domain not in constraint_dict:
						constraint_dict[domain] = {}

					slot = token.split('_')[1]

					constraint_dict[domain][slot] = []
				else:
					domain = token
			else:
				try:
					constraint_dict[domain][slot].append(token)
				except KeyError:
					continue

		for domain, sv_dict in constraint_dict.items():
			for s, value_tokens in sv_dict.items():
				constraint_dict[domain][s] = ' '.join(value_tokens)
		return constraint_dict

	def convert_format(self, results):
		converted_results: Dict[str, Dict] = defaultdict(list)
		for dial_id, dial in results.items():
			dial_id = dial_id.split('.')[0]
			for turn in dial:
				converted_turn = {'response': ''}
				resp = turn['resp_gen']
				raw_resp = turn.get('resp_raw_gen', None)

				resp = resp.replace('<bos_resp>', '')
				resp = resp.replace('<eos_resp>', '')

				converted_turn['response'] = resp.strip()
				if raw_resp is not None:
					converted_turn['raw_response'] = raw_resp.strip()
				
				# policy_optimization should only be on if trying ppo only without lm
				if not self.is_policy_optimization:
					converted_turn['state'] = self.bspn_to_constraint_dict(turn['bspn_gen'])
				
				converted_results[dial_id].append(converted_turn)
		return converted_results

	def predict_batches(self, pred_batches, keep_raw=False):
		# turn off deterministic for faster generation. being seeded should be enough
		if self.cfg.deterministic:
			torch.use_deterministic_algorithms(False)
		early_stopping = True if self.cfg.beam_size > 1 else False

		results = {}
		for dial_batch in tqdm(pred_batches, total=len(pred_batches), desc="Prediction"):
			batch_size = len(dial_batch)

			dial_history = [[] for _ in range(batch_size)]
			domain_history = [[] for _ in range(batch_size)]
			constraint_dicts = [OrderedDict() for _ in range(batch_size)]
			for turn_batch in self.iterator.transpose_batch(dial_batch):
				batch_encoder_input_ids = []
				for t, turn in enumerate(turn_batch):
					context, _ = self.iterator.flatten_dial_history(
						dial_history[t], [], len(turn["user"]), self.cfg.context_size)

					encoder_input_ids = context + turn["user"] + [self.reader.eos_token_id]

					batch_encoder_input_ids.append(self.iterator.tensorize(encoder_input_ids))

					turn_domain = turn["turn_domain"][-1]

					if "[" in turn_domain:
						turn_domain = turn_domain[1:-1]

					domain_history[t].append(turn_domain)

				batch_encoder_input_ids = pad_sequence(batch_encoder_input_ids,
														batch_first=True,
														padding_value=self.reader.pad_token_id)

				batch_encoder_input_ids = batch_encoder_input_ids.to(self.cfg.device)

				attention_mask = torch.where(
					batch_encoder_input_ids == self.reader.pad_token_id, 0, 1)

				# belief tracking
				with torch.no_grad():
					encoder_outputs = self.model(input_ids=batch_encoder_input_ids,
													attention_mask=attention_mask,
													return_dict=False,
													encoder_only=True,
													add_auxiliary_task=self.cfg.add_auxiliary_task)

					span_outputs, encoder_hidden_states = encoder_outputs

					if isinstance(encoder_hidden_states, tuple):
						last_hidden_state = encoder_hidden_states[0]
					else:
						last_hidden_state = encoder_hidden_states

					# wrap up encoder outputs
					encoder_outputs = BaseModelOutput(
						last_hidden_state=last_hidden_state)

					belief_outputs = self.model.generate(encoder_outputs=encoder_outputs,
															attention_mask=attention_mask,
															eos_token_id=self.reader.eos_token_id,
															max_new_tokens=75,
															do_sample=self.cfg.do_sample,
															num_beams=self.cfg.beam_size,
															early_stopping=early_stopping,
															temperature=self.cfg.temperature,
															top_k=self.cfg.top_k,
															top_p=self.cfg.top_p,
															repetition_penalty=self.cfg.repetition_penalty,
															decoder_type="belief")

				belief_outputs = belief_outputs.cpu().numpy().tolist()

				# span prediction
				pred_spans = span_outputs[1].cpu().numpy().tolist()
				input_ids = batch_encoder_input_ids.cpu().numpy().tolist()

				decoded_belief_outputs = self.finalize_bspn(
					belief_outputs, domain_history, constraint_dicts, pred_spans, input_ids)

				for t, turn in enumerate(turn_batch):
					turn.update(**decoded_belief_outputs[t])

				dbpn = []
				bspn = []
				aspn = []
				# compute database pointer, which will be used for generation
				for turn in turn_batch:
					# if policy optimization, use the ground truth db pointer
					if self.is_policy_optimization:
						bspn_gen = turn["bspn"]
						bspn.append(bspn_gen)

						dbpn_gen = turn["dbpn"]
						turn["dbpn_gen"] = dbpn_gen
						dbpn.append(dbpn_gen)
					else:
						if self.cfg.add_auxiliary_task:
							bspn_gen = turn["bspn_gen_with_span"]
						else:
							bspn_gen = turn["bspn_gen"]

						bspn.append(bspn_gen)
						bspn_gen = self.reader.tokenizer.decode(
							bspn_gen, clean_up_tokenization_spaces=False)

						db_token = self.reader.bspn_to_db_pointer(bspn_gen, turn["turn_domain"])

						dbpn_gen = self.reader.encode_text(
							db_token,
							bos_token=definitions.BOS_DB_TOKEN,
							eos_token=definitions.EOS_DB_TOKEN)

						turn["dbpn_gen"] = dbpn_gen
						dbpn.append(dbpn_gen)
					aspn.append(turn["aspn"])  # used only for policy optimization

				if self.cfg.resp_use_bspn:
					# clean up bspn
					bos_belief_token_id: int = self.reader.get_token_id(definitions.BOS_BELIEF_TOKEN)
					eos_belief_token_id: int = self.reader.get_token_id(definitions.EOS_BELIEF_TOKEN)
					for bs in bspn:
						if bos_belief_token_id not in bs:
							bs.insert(0, bos_belief_token_id)
						if eos_belief_token_id not in bs:
							bs.append(eos_belief_token_id)
					
					# resp input is PADDING + bspn + dbpn
					for t, (bs, db) in enumerate(zip(bspn, dbpn)):
						dbpn[t] = bs + db
					seq_max_length = max([len(s) for s in dbpn])
					for t, s in enumerate(dbpn):
						# T5 use pad_token as start_decoder_token_id
						curr_len = len(s)
						dbpn[t] = [self.reader.pad_token_id] * (seq_max_length - curr_len) + s
				else:
					# resp input is PADDING + dbpn
					for t, db in enumerate(dbpn):
						dbpn[t] = [self.reader.pad_token_id] + db
				
				# seperate act and resp generation
				if self.cfg.sep_act_n_resp_gen:
					bos_act_token_id: int = self.reader.get_token_id(definitions.BOS_ACTION_TOKEN)
					eos_act_token_id: int = self.reader.get_token_id(definitions.EOS_ACTION_TOKEN)
					bos_resp_token_id: int = self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
					eos_resp_token_id: int = self.reader.get_token_id(definitions.EOS_RESP_TOKEN)

					if self.is_policy_optimization and self.cfg.use_true_curr_aspn:
						for t, db in enumerate(dbpn):
							dbpn[t] = db + aspn[t]
					else:
						for t, db in enumerate(dbpn):
							dbpn[t] = db + [bos_act_token_id]
						resp_decoder_input_ids = self.iterator.tensorize(dbpn)
						resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

						# act generation
						with torch.no_grad():
							act_outputs = self.model.generate(
								encoder_outputs=encoder_outputs,
								attention_mask=attention_mask,
								decoder_input_ids=resp_decoder_input_ids,
								pad_token_id=self.reader.pad_token_id,
								eos_token_id=self.reader.eos_token_id,
								max_new_tokens=20,
								do_sample=self.cfg.do_sample,
								num_beams=self.cfg.beam_size,
								early_stopping=early_stopping,
								temperature=self.cfg.temperature,
								top_k=self.cfg.top_k,
								top_p=self.cfg.top_p,
								repetition_penalty=self.cfg.repetition_penalty,
								decoder_type="resp")

						act_outputs = act_outputs.cpu().numpy().tolist()

						# clean act outputs
						for t, act in enumerate(act_outputs):
							if eos_act_token_id not in act:
								cleaned_act: list[int] = [eos_act_token_id]
							else:
								cleaned_act = act[act.index(bos_act_token_id)+1:act.index(eos_act_token_id)+1]
							dbpn[t] = dbpn[t] + cleaned_act
					
					# pad left and add bos_resp_token_id
					seq_max_length = max([len(s) for s in dbpn])
					for t, s in enumerate(dbpn):
						# T5 use pad_token as start_decoder_token_id
						curr_len = len(s)
						dbpn[t] = [self.reader.pad_token_id] * (seq_max_length - curr_len) + s + [bos_resp_token_id]
					
					# resp generation
					resp_decoder_input_ids = self.iterator.tensorize(dbpn)
					resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

					# response generation
					with torch.no_grad():
						resp_outputs = self.model.generate(
							encoder_outputs=encoder_outputs,
							attention_mask=attention_mask,
							decoder_input_ids=resp_decoder_input_ids,
							pad_token_id=self.reader.pad_token_id,
							eos_token_id=self.reader.eos_token_id,
							max_new_tokens=100,
							do_sample=self.cfg.do_sample,
							num_beams=self.cfg.beam_size,
							early_stopping=early_stopping,
							temperature=self.cfg.temperature,
							top_k=self.cfg.top_k,
							top_p=self.cfg.top_p,
							repetition_penalty=self.cfg.repetition_penalty,
							decoder_type="resp")
					resp_outputs = resp_outputs.cpu().numpy().tolist()
					# add eos_resp_token_id
					for t, resp in enumerate(resp_outputs):
						if eos_resp_token_id not in resp:
							resp_outputs[t] = resp + [eos_resp_token_id]
				else:
					# generate resp and act together
					# aspn has different length
					resp_decoder_input_ids = self.iterator.tensorize(dbpn)
					resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

					# response generation
					with torch.no_grad():
						resp_outputs = self.model.generate(
							encoder_outputs=encoder_outputs,
							attention_mask=attention_mask,
							decoder_input_ids=resp_decoder_input_ids,
							eos_token_id=self.reader.eos_token_id,
							max_length=300,
							do_sample=self.cfg.do_sample,
							num_beams=self.cfg.beam_size,
							early_stopping=early_stopping,
							temperature=self.cfg.temperature,
							top_k=self.cfg.top_k,
							top_p=self.cfg.top_p,
							repetition_penalty=self.cfg.repetition_penalty,
							decoder_type="resp")

					resp_outputs = resp_outputs.cpu().numpy().tolist()

				decoded_resp_outputs = self.finalize_resp(resp_outputs, keep_raw=keep_raw)

				for t, turn in enumerate(turn_batch):
					turn.update(**decoded_resp_outputs[t])
				
				# update dial_history
				for t, turn in enumerate(turn_batch):
					pv_text = copy.copy(turn["user"])

					pv_bspn = turn["bspn_gen_with_span"]
					pv_dbpn = turn["dbpn_gen"]
					pv_aspn = turn["aspn_gen"]
					pv_resp = turn["resp_gen"]
					pv_text += (pv_bspn + pv_dbpn + pv_aspn + pv_resp)

					dial_history[t].append(pv_text)

			result = self.iterator.get_readable_batch(dial_batch)
			results.update(**result)

		converted_results = self.convert_format(results)
		if self.cfg.deterministic:
			torch.use_deterministic_algorithms(True)
		return converted_results

	def predict(self):
		self.model.eval()

		iterator = self.iterator
		if self.test_iterator is not None:
			iterator = self.test_iterator
		
		pred_batches, _, _, _ = iterator.get_batches(
			self.cfg.pred_data_type, self.cfg.batch_size,
			self.cfg.num_gpus, excluded_domains=self.cfg.excluded_domains)

		if self.cfg.debug or self.cfg.pilot_run:
			pred_batches = pred_batches[:5]
		
		with torch.no_grad():
			results = self.predict_batches(pred_batches)

		# evaluate
		score = self.__evaluator.evaluate(copy.deepcopy(results))  # evaluate will modify results
		print(json.dumps(score, indent=4))
		total_score = score['bleu']['mwz22'] + (score['success']['inform']['total'] + score['success']['success']['total']) / 2.0
		print(f"Total score: {total_score:.3f}")

		# save
		if self.cfg.output:  # happens during evaluation
			save_json(results, self.cfg.output)
		elif self.cfg.save_val_predictions and self.cfg.model_dir is not None:  # happens during training
			latest_ckpt = "test_predictions.json"
			save_dir = os.path.join(self.cfg.model_dir, "preds")
			Path(save_dir).mkdir(parents=True, exist_ok=True)

			save_path = os.path.join(save_dir, latest_ckpt)
			perf = {'predictions': results, 'score': score}
			save_json(perf, save_path)
		return