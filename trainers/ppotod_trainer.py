import torch
import numpy as np
import os
import json
import copy
import re
import pickle
import ast

from typing import *
from tqdm.auto import tqdm
from models.ppotod import PPOTODModel

from pathlib import Path
from deprecated import deprecated
from data_utils.mttod.utils import definitions
from trainers.mttod_trainer import MTTODTrainer
from transformers import TopPLogitsWarper, TemperatureLogitsWarper
from torch.distributions import Categorical
from torchmetrics.functional import bleu_score
from bert_score import BERTScorer
from utils.reporter import WandbReporter, SimpleReporter
from utils.utils import seed_everything, cfg_to_dict, get_or_create_logger, get_mean_std_with_padding
from utils.db import MyDB


logger = get_or_create_logger(__name__)


class PPOLMTrainer(MTTODTrainer):
	def __init__(self, dataset, cfg, model:PPOTODModel, group_name, run_name, test_dataset=None, ray_reporter=None, kl_model=None):
		super(PPOLMTrainer, self).__init__(dataset, cfg, model, test_dataset=test_dataset)
		if cfg.mode == 'train':
			self.reporter: SimpleReporter = WandbReporter(cfg, group_name, run_name)
			self.rl_reporter: SimpleReporter = WandbReporter(cfg, group_name, run_name, run=self.reporter.run)
			self.hidden_reporter: SimpleReporter = WandbReporter(cfg, group_name, run_name, run=self.reporter.run)  # for intermediate hidden states
			self.val_reporter: SimpleReporter = WandbReporter(cfg, group_name, run_name, run=self.reporter.run)
		else:
			# simple rl_reporter already initialized in base class
			self.rl_reporter = SimpleReporter(cfg)
			self.hidden_reporter = SimpleReporter(cfg)
			self.val_reporter = SimpleReporter(cfg) # global_step%logging_freq should never trigger logging
		self.val_reporter.log_frequency = float("inf")
		self.ray_reporter = ray_reporter

		self.kl_model = kl_model
		is_policy_optimization = self.cfg.is_policy_optimization
		self.is_policy_optimization = is_policy_optimization if is_policy_optimization is not None else (self.cfg.use_ppo and not self.cfg.use_sl)
		if self.cfg.use_bert_score:
			self._bert_scorer = BERTScorer(
				model_type="roberta-base", lang="en",
				all_layers=False,
				rescale_with_baseline=True, 
				device=self.cfg.device
			)
		if self.cfg.reward == "token_sim":
			# load embedding matrix
			logger.info(f"loading embedding matrix from {self.cfg.token_embedding_path}")
			self.__token_embedding = torch.load(self.cfg.token_embedding_path)
		
		self.__cfg_backups: Dict[str, Any] = {}
		self.__check_configs()

		self.db = None
		self.__init_dbs()  # for saving some precomputed stuff like KL logits. But in the end local disk explodes as its big

		self.__freeze()
		return

	def __freeze(self):
		if self.cfg.freeze_encoder is None or not self.cfg.freeze_encoder:
			return
		
		for param in self.model.model.encoder.parameters():
			param.requires_grad = False
		logger.info("encoder is frozen")
		return

	def __check_configs(self):
		logger.info("model received:")
		self.cfg.token_prob_temperature = float(self.cfg.token_prob_temperature)
		self.cfg.token_prob_scale = float(self.cfg.token_prob_scale)
		self.cfg.alternate_step_k = float(self.cfg.alternate_step_k)
		self.cfg.correct_reward = float(self.cfg.correct_reward)
		logger.info(json.dumps(cfg_to_dict(self.cfg), indent=4))

		if self.cfg.add_gold_demonstrations and self.cfg.use_gold_prob > 0.0:
			logger.warn("add_gold_demonstrations set to True and use_gold_prob > 0.0")
		if self.cfg.reward == "zeros" and not self.cfg.add_terminal_reward:
			logger.warn("nonterminal reward are zeros AND add_terminal_reward is False")
		if self.cfg.alternate_epoch == True and self.cfg.alternate_step_k > 0:
			logger.warn(f"alternate_epoch is True but {self.cfg.alternate_step_k=} is larger than 0")
		if self.cfg.alternate_step_k > 0 and self.cfg.update_old_policy_interval > 1:
			logger.warn(f"{self.cfg.alternate_step_k=} is larger than 0 but {self.cfg.update_old_policy_interval=}")
		if (self.cfg.debug or self.cfg.pilot_run) and self.cfg.deterministic:
			logger.warn("it is suggested to use non-deterministic training (faster) for debugging or pilot run")
		if self.cfg.mode == "predict" and self.cfg.deterministic:
			logger.warn("it is suggested to use non-deterministic training (faster) for prediction")
		if self.cfg.sample_action == "greedy" and self.cfg.num_per_sample > 1:
			logger.warn(f"using greedy sampling but number of actions to sampler per data point is {self.cfg.num_per_sample=}")
		# storage explodes
		# if ((self.cfg.add_kl_divergence and not self.cfg.precompute_kl_logits) 
		# 		or (self.cfg.reward == "token_prob" and not self.cfg.precompute_token_prob_logits)):
		# 	logger.warn("it is recommended to use db for precomputing KL and token sim logits for faster training")
		if self.is_policy_optimization and self.cfg.alternate_step_k > 0 and self.cfg.alternate_epoch:
			raise ValueError(f"{self.cfg.alternate_step_k=} is larger than 0 and policy optimization (i.e. RL only) is used, but alternate_epoch is deprecated")
		if self.cfg.num_per_sample > 1 and self.cfg.alternate_epoch:
			raise NotImplementedError("num_per_sample > 1 and alternate_epoch are not implemented yet")
		if self.cfg.reward == "token_prob" and self.kl_model is None:
			raise ValueError("token_prob reward requires a trained 'kl_model'. To load it you can specify --add_kl_divergence. To not penalize kl set kl_scale to 0.0")
		if self.cfg.lm_head_model_action_value and self.cfg.adv_use_returns:
			raise ValueError(f"{self.cfg.adv_use_returns=} disables value computation, but {self.cfg.lm_head_model_action_value=} needs value")
		if self.is_policy_optimization and self.cfg.precompute_kl_logits:
			raise ValueError("policy optimization (i.e. RL only) currently does not support precomputing KL logits")
		logger.info(f"{self.is_policy_optimization=}")
		return

	def __init_dbs(self):
		if ((self.cfg.add_kl_divergence and self.cfg.precompute_kl_logits) 
				or (self.cfg.reward == "token_prob" and self.cfg.precompute_token_prob_logits)):
			db_path = self.cfg.precompute_db_path
			self.db = MyDB(db_path, curr_batch_size=self.cfg.batch_size)
			logger.info(f"loading db at {db_path}")
		return

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

	def get_resp_attention_mask(self, resp_label_ids):
		# we count DA as part of the response
		if self.cfg.penalize_da_tokens:
			bos_resp_token_id = self.reader.get_token_id(definitions.BOS_ACTION_TOKEN)
		else:
			bos_resp_token_id = self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
		eos_resp_token_id = self.reader.get_token_id(definitions.EOS_RESP_TOKEN)
		return self.get_attention_mask_span(resp_label_ids, bos_resp_token_id, eos_resp_token_id)

	def compute_reward(self, resp_pred, resp_label, reward_shape, batch_idx, is_gold):
		reward_shape = resp_pred.shape
		if self.cfg.reward == "zeros":
			# terminal reward is added later
			return np.zeros(resp_pred.shape)

		all_special_tokens = self.model.model._special_token_ids # + self.model.model._da_token_ids
		special_token_mask = np.isin(resp_label, all_special_tokens)
		bos_response_mask = resp_label == self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
		eos_response_mask = resp_label == self.reader.get_token_id(definitions.EOS_RESP_TOKEN)

		unnormed_reward = np.zeros(resp_pred.shape)

		if is_gold:  # faster
			reward = np.ones(reward_shape)
			reward += special_token_mask * (self.cfg.special_token_error_scale - 1.0)
			unnormed_reward = reward.copy()

			reward = (reward - np.min(reward)) / (np.max(reward) - np.min(reward) + 1e-6)
			reward = reward * (self.cfg.correct_reward + 1.0) - 1.0
		elif self.cfg.reward == "sentence_error":
			reward = -1.0 * np.sum(special_token_mask * (resp_pred != resp_label), axis=1, keepdims=True)
			reward = reward * np.ones(reward_shape)
		elif self.cfg.reward == "token_error":
			reward = np.zeros(reward_shape)
			attention_mask = np.where(resp_label == self.reader.pad_token_id, 0, 1)
			reward += -1.0 * (attention_mask * (resp_pred != resp_label)) # wrong word
			reward += 1.0 * (attention_mask * (resp_pred == resp_label)) # correct word
			special_token_scale = self.cfg.special_token_error_scale - 1.0
			reward += -1.0 * special_token_scale * (special_token_mask * (resp_pred != resp_label)) # wrong special token
			reward += 1.0 * special_token_scale * (special_token_mask * (resp_pred == resp_label)) # correct special token

			unnormed_reward = reward.copy()
			reward = (reward - np.min(reward)) / (np.max(reward) - np.min(reward) + 1e-6)
			reward = reward * (self.cfg.correct_reward + 1.0) - 1.0
		elif self.cfg.reward == "token_sim":
			torch_resp_pred = torch.from_numpy(resp_pred).to(self.__token_embedding.weight.device)
			torch_resp_label = torch.from_numpy(resp_label).to(self.__token_embedding.weight.device)
			
			resp_preds_embedding = self.__token_embedding(torch_resp_pred)
			resp_labels_embedding = self.__token_embedding(torch_resp_label)

			cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
			reward = cos_sim(resp_preds_embedding, resp_labels_embedding).detach().cpu().numpy()

			resp_attention_mask = self.get_attention_mask_span(
				torch.tensor(resp_label),
				self.reader.get_token_id(definitions.BOS_RESP_TOKEN), self.reader.get_token_id(definitions.EOS_RESP_TOKEN), 
				include_token_id=False
			).numpy()
			reward = reward * resp_attention_mask

			# for special tokens, we assign manually as we need to get it exactly right
			incorrect_special_token_mask = special_token_mask * (resp_pred != resp_label)
			correct_special_token_mask = special_token_mask * (resp_pred == resp_label)
			
			reward[incorrect_special_token_mask] = -1.0 * self.cfg.special_token_error_scale # wrong special token
			reward[correct_special_token_mask] = 1.0 * self.cfg.special_token_error_scale # correct special token

			unnormed_reward = reward.copy()
			reward = (reward - np.min(reward)) / (np.max(reward) - np.min(reward) + 1e-6)
			reward = reward * (self.cfg.correct_reward + 1.0) - 1.0
		elif self.cfg.reward == "token_contextual_sim":
			torch_resp_pred = torch.from_numpy(resp_pred).to(self.model.model.device)
			torch_resp_label = torch.from_numpy(resp_label).to(self.model.model.device)
			# get the output probabilities
			if self.db is None:
				with torch.no_grad():
					attention_mask = torch.where(torch_resp_label == self.reader.pad_token_id, 0, 1)
					# used for both kl and here
					ref_decoder_output, *_ = self.kl_model.resp_forward(input_ids=torch_resp_label,	
													attention_mask=attention_mask,
													encoder_outputs=None,
													lm_labels=torch_resp_label,
													return_decoder_outputs=True,
													return_dict=False)
					ref_decoder_output = ref_decoder_output[0]  # last layer, i.e. 11-th layer in BERTScore

					pred_decoder_output, *_ = self.kl_model.resp_forward(input_ids=torch_resp_pred,	
													attention_mask=attention_mask,
													encoder_outputs=None,
													lm_labels=torch_resp_pred,
													return_decoder_outputs=True,
													return_dict=False)
					pred_decoder_output = pred_decoder_output[0]  # last layer, i.e. 11-th layer in BERTScore
			else:
				ref_decoder_output = self.db.search('kl_decoder_output', batch_idx)
				ref_decoder_output = torch.from_numpy(ref_decoder_output).to(self.model.model.device)
				pred_decoder_output = self.db.search('kl_decoder_output', batch_idx)
				pred_decoder_output = torch.from_numpy(pred_decoder_output).to(self.model.model.device)

			cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
			reward = cos_sim(pred_decoder_output, ref_decoder_output).detach().cpu().numpy()

			# for special tokens, we assign manually as we need to get it exactly right
			incorrect_special_token_mask = special_token_mask * (resp_pred != resp_label)
			correct_special_token_mask = special_token_mask * (resp_pred == resp_label)
			
			reward[incorrect_special_token_mask] = -1.0 * self.cfg.special_token_error_scale # wrong special token
			reward[correct_special_token_mask] = 1.0 * self.cfg.special_token_error_scale # correct special token

			unnormed_reward = reward.copy()
			reward = (reward - np.min(reward)) / (np.max(reward) - np.min(reward) + 1e-6)
			reward = reward * (self.cfg.correct_reward + 1.0) - 1.0
		elif self.cfg.reward == "token_prob":
			torch_resp_pred = torch.from_numpy(resp_pred).to(self.model.model.device)
			torch_resp_label = torch.from_numpy(resp_label).to(self.model.model.device)
			# get the output probabilities
			if self.db is None:
				with torch.no_grad():
					attention_mask = torch.where(torch_resp_label == self.reader.pad_token_id, 0, 1)
					# used for both kl and here
					ref_logit, _, _ = self.kl_model.resp_forward(input_ids=torch_resp_label,	
													attention_mask=attention_mask,
													encoder_outputs=None,
													lm_labels=torch_resp_label,
													return_dict=False)
			else:
				ref_logit = self.db.search('kl_logits', batch_idx)
				ref_logit = torch.from_numpy(ref_logit).to(self.model.model.device)
			# reshape the distribution
			temp_wrapper = TemperatureLogitsWarper(temperature=self.cfg.token_prob_temperature)
			ref_logit = temp_wrapper(None, ref_logit)

			ref_probs = torch.softmax(ref_logit, dim=-1)
			sim_score = torch.gather(ref_probs, dim=-1, index=torch_resp_pred.unsqueeze(-1)).squeeze(-1)
			
			# reshape to -1 and 1
			reward = sim_score.detach().cpu().numpy()
			reward *= self.cfg.token_prob_scale  # to -1.0 * scale, 1.0 * scale, 0.0 < scale <= 1.0

			# for correct non-special tokens
			correct_nonspecial_token_mask = (resp_pred == resp_label) * ~special_token_mask
			reward[correct_nonspecial_token_mask] = 1.0
			
			# for special tokens
			incorrect_special_token_mask = special_token_mask * (resp_pred != resp_label)
			correct_special_token_mask = special_token_mask * (resp_pred == resp_label)
			reward[incorrect_special_token_mask] = -1.0 * self.cfg.special_token_error_scale # wrong special token
			reward[correct_special_token_mask] = 1.0 * self.cfg.special_token_error_scale # correct special token

			unnormed_reward = reward.copy()
			reward = (reward - np.min(reward)) / (np.max(reward) - np.min(reward) + 1e-6)
			reward = reward * (self.cfg.correct_reward + 1.0) - 1.0

		elif self.cfg.reward == "token_confidence":
			torch_resp_pred = torch.from_numpy(resp_pred).to(self.model.model.device)
			torch_resp_label = torch.from_numpy(resp_label).to(self.model.model.device)

			# get the output probabilities
			with torch.no_grad():
				attention_mask = torch.where(torch_resp_label == self.reader.pad_token_id, 0, 1)
				# used for both kl and here
				ref_logit, _, _ = self.model.resp_forward(input_ids=torch_resp_pred,	
												attention_mask=attention_mask,
												encoder_outputs=None,
												lm_labels=torch_resp_label,
												return_dict=False)

			ref_probs = torch.softmax(ref_logit, dim=-1)
			lm_preds_probs = torch.gather(ref_probs, dim=-1, index=torch_resp_pred.unsqueeze(-1)).squeeze(-1)
			lm_preds_probs = lm_preds_probs.detach().cpu().numpy()

			# reshape to -1 and 1
			reward = np.zeros_like(lm_preds_probs)
			confidence = lm_preds_probs

			# for correct non-special tokens
			incorrect_nonspecial_token_mask = (resp_pred != resp_label) * ~special_token_mask
			correct_nonspecial_token_mask = (resp_pred == resp_label) * ~special_token_mask
			reward += self.cfg.correct_reward * confidence * correct_nonspecial_token_mask
			reward += -1.0 * np.clip(confidence, a_min=0.1, a_max=None) * incorrect_nonspecial_token_mask
			
			# for special tokens
			incorrect_special_token_mask = special_token_mask * (resp_pred != resp_label)
			correct_special_token_mask = special_token_mask * (resp_pred == resp_label)
			reward += (self.cfg.correct_reward * self.cfg.special_token_error_scale) * confidence * correct_special_token_mask
			reward += (-1.0 * self.cfg.special_token_error_scale) * np.clip(confidence, a_min=0.1, a_max=None) * incorrect_special_token_mask

			# remap 
			unnormed_reward = reward.copy()
			reward = (reward - np.min(reward)) / (np.max(reward) - np.min(reward) + 1e-6)
			reward = reward * (self.cfg.correct_reward + 1.0) - 1.0
		# make sure the reward is correct
		resp_attention_mask = self.get_attention_mask_span(
			torch.tensor(resp_label),
			self.reader.get_token_id(definitions.BOS_RESP_TOKEN), self.reader.get_token_id(definitions.EOS_RESP_TOKEN), 
			include_token_id=False
		).numpy()
		reward = reward * resp_attention_mask
		
		unnormed_reward *= resp_attention_mask
		if self.hidden_reporter is not None:
			self.hidden_reporter.step({"unnorm_average_reward": unnormed_reward.sum() / resp_attention_mask.sum()})

		# has to be somewhere because we assumed eos_resp is generated correctly
		reward += -1.0 * bos_response_mask * (resp_pred != resp_label)
		reward += self.cfg.correct_reward * bos_response_mask * (resp_pred == resp_label)
		reward += -1.0 * eos_response_mask * (resp_pred != resp_label)
		reward += self.cfg.correct_reward * eos_response_mask * (resp_pred == resp_label)
		
		# compute act reward
		if self.cfg.penalize_da_tokens:
			bos_act_token_id = self.reader.get_token_id(definitions.BOS_ACTION_TOKEN)
			eos_act_token_id = self.reader.get_token_id(definitions.EOS_ACTION_TOKEN)
			act_span_mask = self.get_attention_mask_span(torch.tensor(resp_label), bos_act_token_id, eos_act_token_id, include_token_id=False)
			act_span_mask = act_span_mask.numpy()
			special_token_scale = self.cfg.special_token_error_scale - 1.0
			reward += self.cfg.correct_reward * special_token_scale * act_span_mask * (resp_pred == resp_label)  # correct
			reward += -1.0 * special_token_scale * act_span_mask * (resp_pred != resp_label)  # wrong
			# penalize ths special tokens
			bos_act_mask = resp_label == self.reader.get_token_id(definitions.BOS_ACTION_TOKEN)
			eos_act_mask = resp_label == self.reader.get_token_id(definitions.EOS_ACTION_TOKEN)
			reward += -1.0 * bos_act_mask * (resp_pred != resp_label)
			reward += self.cfg.correct_reward * bos_act_mask * (resp_pred == resp_label)
			reward += -1.0 * eos_act_mask * (resp_pred != resp_label)
			reward += self.cfg.correct_reward * eos_act_mask * (resp_pred == resp_label)
		return reward

	def compute_special_token_f1(self, single_pred, single_label):
		resp_special_token_ids = self.model.model._special_token_ids
		pred_sp_id_mask = np.isin(single_pred, resp_special_token_ids)
		label_sp_id_mask = np.isin(single_label, resp_special_token_ids)
		pred_sp_ids = single_pred[pred_sp_id_mask]
		label_sp_ids = single_label[label_sp_id_mask]
		# precision
		if len(pred_sp_ids) == 0:
			precision = 1.0
		else:
			num_correct = np.isin(pred_sp_ids, label_sp_ids).sum()
			precision = num_correct / len(pred_sp_ids)
		# recall
		if len(label_sp_ids) == 0:
			recall = 1.0
		else:
			num_correct = np.isin(label_sp_ids, pred_sp_ids).sum()
			recall = num_correct / len(label_sp_ids)
		# f1
		if precision + recall == 0:
			f1 = 0.0
		else:
			f1 = 2 * precision * recall / (precision + recall)
		return f1

	def compute_bert_score_reward(self, resp_pred, resp_label):
		decoded_resp_pred = self.reader.tokenizer.decode(resp_pred, skip_special_tokens=False)
		decoded_resp_label = self.reader.tokenizer.decode(resp_label, skip_special_tokens=False)

		pred = self.reader.ensure_space_between_special_symbols(decoded_resp_pred)
		label = self.reader.ensure_space_between_special_symbols(decoded_resp_label)
		cleaned_pred = re.sub(r'\[value_(.*?)\]', r'\1', pred)
		cleaned_pred = re.sub(r'\[(.*?)\]', r'\1', cleaned_pred)  # e.g. no_offer to no_offer
		cleaned_label = re.sub(r'\[value_(.*?)\]', r'\1', label)

		score = self._bert_scorer.score([cleaned_pred], [cleaned_label])
		return score[-1][0]  # f1

	def compute_terminal_reward(self, resp_pred, resp_label, reward):
		bos_resp_token_id = self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
		eos_resp_token_id = self.reader.get_token_id(definitions.EOS_RESP_TOKEN)
		start_id = np.where(resp_label == bos_resp_token_id)[1]
		end_id = np.where(resp_label == eos_resp_token_id)[1]
		for i, (sid, eid) in enumerate(zip(start_id, end_id)):
			pred_sentence = resp_pred[i, sid+1: eid]
			label_sentence = resp_label[i, sid+1: eid]

			# bleu calculates n-gram precision, so it does not need to know the content
			fake_pred_sent = ' '.join([str(x) for x in pred_sentence])
			fake_label_sent = ' '.join([str(x) for x in label_sentence])
			bleu = bleu_score([fake_pred_sent], [fake_label_sent], n_gram=3)
			# special_token_f1 approximates inform and success
			special_token_f1 = self.compute_special_token_f1(pred_sentence, label_sentence)
			if self.cfg.terminal_reward_fn == "all":
				terminal_reward = bleu + special_token_f1
			elif self.cfg.terminal_reward_fn == "sp_f1":
				terminal_reward = special_token_f1
			
			if self.cfg.use_bert_score:
				# TODO: ideally this should be a one-to-one mapping of score to each token
				bert_score = self.compute_bert_score_reward(pred_sentence, label_sentence)
				terminal_reward += bert_score

			# add the reward to the eos_resp token of label
			reward[i, eid] += self.cfg.rl_gamma * self.cfg.terminal_reward_scale * terminal_reward
		return reward

	def compute_advantage_n_returns(self, 
			greedy_preds: torch.Tensor, 
			resp_labels: torch.Tensor,
			values: torch.Tensor,
			batch_idx: int,
			is_gold=False):
		# collect advantage
		resp_pred = greedy_preds.detach().cpu().numpy()
		resp_label = resp_labels.detach().cpu().numpy()

		reward: np.ndarray = self.compute_reward(resp_pred, resp_label, reward_shape=values.shape, batch_idx=batch_idx, is_gold=is_gold)
		if self.cfg.add_terminal_reward:
			reward = self.compute_terminal_reward(resp_pred, resp_label, reward)

		# calcuate delta_t = r_t + gamma * V_{t+1} - V_t
		advantages_arr: np.ndarray = np.zeros_like(reward)
		if not self.cfg.adv_use_returns:
			values_t1 = torch.concat([values[..., 1:], torch.zeros_like(values[..., -1:])], dim=-1)
			values_t1 = values_t1.detach().cpu().numpy()
			values = values.detach().cpu().numpy()
			delta_t = reward + (self.cfg.rl_gamma * values_t1) - values
			# compute A_t = delta_t + (gamma * lambda) * delta_{t+1} + ... + (gamma * lambda) ^ (T - t) * delta_T
			# compute from the end
			advantages: List[np.ndarray] = []
			for t in range(delta_t.shape[1] - 1, -1, -1):
				A_t = np.sum(delta_t[:, t:], axis=1, keepdims=True)
				advantages.insert(0, A_t)
				delta_t[:, t:] *= self.cfg.rl_gamma * self.cfg.rl_lambda
			advantages_arr = np.concatenate(advantages, axis=1)
			if self.cfg.normalize_advantage:
				advantages_arr = (advantages_arr - np.mean(advantages_arr)) / (np.std(advantages_arr) + 1e-8)

		# compute returns. since we only care about resp, others NEED to be masked
		reward_copy = np.copy(reward)
		resp_attention_mask = self.get_resp_attention_mask(resp_labels).cpu().numpy()
		reward_copy *= resp_attention_mask

		returns: List[np.ndarray] = []
		for t in range(reward_copy.shape[1] - 1, -1, -1):
			R_t = np.sum(reward_copy[:, t:], axis=1, keepdims=True)
			returns.insert(0, R_t)
			reward_copy[:, t:] *= self.cfg.rl_gamma
		returns_arr: np.ndarray = np.concatenate(returns, axis=1)
		returns_arr *= resp_attention_mask  # remove unneeded rewards
		if self.cfg.real_normalize_return:
			# intuitively this would penalize half of the actions, but practically hard to interpret what is going on
			mean, std = get_mean_std_with_padding(returns_arr, 0)
			returns_arr = (returns_arr - mean) / (std + 1e-8)
		elif self.cfg.normalize_return:
			returns_arr = returns_arr / resp_attention_mask.sum().item()
		returns_arr *= resp_attention_mask

		return reward, advantages_arr, returns_arr

	def sample_action(self, lm_logits: torch.Tensor, num_per_sample: int = 1) -> List[torch.Tensor]:
		if self.cfg.sample_action == 'greedy':
			greedy_preds = torch.argmax(lm_logits, dim=-1)
			return [greedy_preds] * num_per_sample
		elif self.cfg.sample_action == 'sample':
			torch_actions = []
			for i in range(num_per_sample):
				actions = []
				for i in range(lm_logits.shape[0]):
					# does not have deterministic implementation
					if self.cfg.deterministic:
						torch.use_deterministic_algorithms(False)
						top_p_wrapper = TopPLogitsWarper(top_p=self.cfg.sample_top_p)
						new_logit = top_p_wrapper(None, lm_logits[i])
						torch.use_deterministic_algorithms(True)
					else:
						top_p_wrapper = TopPLogitsWarper(top_p=self.cfg.sample_top_p)
						new_logit = top_p_wrapper(None, lm_logits[i])

					temp_wrapper = TemperatureLogitsWarper(temperature=self.cfg.sample_temperature)
					new_logit = temp_wrapper(None, new_logit)
					exp_probs = torch.softmax(new_logit, dim=-1)
					dist = Categorical(exp_probs)
					action = dist.sample()
					actions.append(action)
				torch_actions.append(torch.stack(actions, dim=0)) 
			return torch_actions
		else:
			raise ValueError('sample_action must be greedy or sample')

	def collect_rollout(self, data_iterator, num_steps_per_epoch, num_per_sample=1, epoch=-1):
		"""collect experience using current policy, i.e. reponses using greedy decoding generated by current policy
		1) collect the greedy decoded action/token_ids
		2) collect their probabilities
		3) collect their TD(lambda) returns (used for training the value function)
		4) collect the advantage per word = r(s,t) + V(s,t+1) - V(s,t)

		Args:
			data_iterator (_type_): _description_
		"""
		_input_resp_labels = []
		collected_token_ids = []
		collected_probs = []
		collected_rewards = []
		collected_returns = []
		collected_advantages = []

		pbar = tqdm(total=num_steps_per_epoch, desc="collecting rollout")
		for step, batch in enumerate(data_iterator):
			inputs, (_, _, resp_labels) = batch
			_input_resp_labels.append(resp_labels)

			inputs = inputs.to(self.cfg.device)
			resp_labels = resp_labels.to(self.cfg.device)
			attention_mask = torch.where(inputs == self.reader.pad_token_id, 0, 1)
			with torch.no_grad():
				lm_logits, values, action_values = self.model.resp_forward(input_ids=inputs,	
															attention_mask=attention_mask,
															encoder_outputs=None,
															lm_labels=resp_labels,
															use_old_model=True,
															return_dict=False)
			
			# collect the token ids
			lm_probs = torch.softmax(lm_logits, dim=-1) # [batch_size, prob_of_each_word]
			if self.cfg.use_gold_prob > 0.0 and np.random.rand() < self.cfg.use_gold_prob:
				lm_preds = [resp_labels]
			else:
				lm_preds = self.sample_action(lm_logits.detach(), num_per_sample=num_per_sample)
			collected_token_ids.append([lm_pred.detach().cpu().numpy() for lm_pred in lm_preds])

			# collect the probabilities
			lm_preds_probs = []
			for lm_pred in lm_preds:
				lm_preds_prob = torch.gather(lm_probs, dim=-1, index=lm_pred.unsqueeze(-1)).squeeze(-1)
				lm_preds_probs.append(lm_preds_prob.detach().cpu().numpy())
			collected_probs.append(lm_preds_probs)

			rewards = []
			advantages = []
			returns = []
			for lm_pred in lm_preds:
				reward, advantage, return_ = self.compute_advantage_n_returns(lm_pred, resp_labels, values, step)
				rewards.append(reward)
				advantages.append(advantage)
				returns.append(return_)
			collected_rewards.append(rewards)
			collected_advantages.append(advantages)
			collected_returns.append(returns)

			if self.cfg.add_gold_demonstrations:
				greedy_preds = resp_labels
				prev_token_ids = collected_token_ids[-1]
				collected_token_ids[-1] = [np.concatenate([prev_token_id, greedy_preds.detach().cpu().numpy()], axis=0) for prev_token_id in prev_token_ids]

				# collect the probabilities
				greedy_preds_probs = torch.gather(lm_probs, dim=-1, index=greedy_preds.unsqueeze(-1)).squeeze(-1)
				prev_probs = collected_probs[-1]
				collected_probs[-1] = [np.concatenate([prev_prob, greedy_preds_probs.detach().cpu().numpy()], axis=0) for prev_prob in prev_probs]

				reward, advantages, returns = self.compute_advantage_n_returns(greedy_preds, resp_labels, values, step)
				prev_rewards = collected_rewards[-1]
				collected_rewards[-1] = [np.concatenate([prev_reward, reward], axis=0) for prev_reward in prev_rewards]
				prev_advantages = collected_advantages[-1]
				collected_advantages[-1] = [np.concatenate([prev_advantage, advantages], axis=0) for prev_advantage in prev_advantages]
				prev_returns = collected_returns[-1]
				collected_returns[-1] = [np.concatenate([prev_return, returns], axis=0) for prev_return in prev_returns]
			
			pbar.update(1)
			if step + 1 == num_steps_per_epoch:
				break
		pbar.close()
		if self.cfg.save_sampled_trajectories:
			latest_ckpt = "sample-at-epoch{}.pkl".format(self._epoch)
			save_dir = os.path.join(self.cfg.model_dir, "preds")
			Path(save_dir).mkdir(parents=True, exist_ok=True)

			save_path = os.path.join(save_dir, latest_ckpt)
			save_data = {'collected_token_ids': collected_token_ids[:1000], '_input_resp_labels': _input_resp_labels[:1000]}
			with open(save_path, 'wb') as f:
				pickle.dump(save_data, f)
		return collected_token_ids, collected_probs, collected_rewards, collected_returns, collected_advantages

	def lm_step_fn(self, *args, **kwargs):
		return self.step_fn(*args, **kwargs)

	def ppo_step_fn(self, resp_input_ids, resp_label_ids, rollout_token_ids, rollout_probs, rollout_rewards, rollout_returns, rollout_advantages, kl_logit, **kwargs):
		"""perform one step of PPO training

		Args:
			resp_input_ids (_type_): _description_
			resp_label_ids (_type_): _description_
			rollout_token_ids (_type_): _description_
			rollout_probs (_type_): _description_
			rollout_advantages (_type_): _description_
		"""
		if self.cfg.add_gold_demonstrations:
			# collected data will have batch_size*2 as half of them is gold
			resp_input_ids = torch.cat([resp_input_ids, resp_input_ids], dim=0)
			resp_label_ids = torch.cat([resp_label_ids, resp_label_ids], dim=0)
		# convert to tensor
		resp_input_ids = resp_input_ids.to(self.cfg.device)
		resp_label_ids = resp_label_ids.to(self.cfg.device)
		rollout_token_ids = torch.tensor(rollout_token_ids, dtype=torch.long, device=self.cfg.device)
		rollout_probs = torch.tensor(rollout_probs, dtype=torch.float, device=self.cfg.device)
		rollout_rewards = torch.tensor(rollout_rewards, dtype=torch.float, device=self.cfg.device)
		rollout_returns = torch.tensor(rollout_returns, dtype=torch.float, device=self.cfg.device)
		rollout_advantages = torch.tensor(rollout_advantages, dtype=torch.float, device=self.cfg.device)

		attention_mask = torch.where(resp_input_ids == self.reader.pad_token_id, 0, 1)
		lm_logits, values, action_values = self.model.resp_forward(input_ids=resp_input_ids,	
												attention_mask=attention_mask,
												encoder_outputs=None,
												lm_labels=resp_label_ids,
												return_dict=False)
		
		resp_attention_mask = self.get_resp_attention_mask(resp_label_ids)
		resp_attention_mask = resp_attention_mask.to(lm_logits.device)

		if self.cfg.adv_use_returns:
			rollout_advantages = rollout_returns
		rollout_advantages = rollout_advantages * resp_attention_mask
		rollout_returns = rollout_returns * resp_attention_mask

		# compute PPO: ratio = pi(a_t|s_t) / pi_old(a_t|s_t) and scaled by advantage
		# compute the log probs
		log_probs = torch.log_softmax(lm_logits, dim=-1)
		current_log_probs = torch.gather(log_probs, dim=-1, index=rollout_token_ids.unsqueeze(-1)).squeeze(-1)
		old_log_probs = torch.log(rollout_probs)

		# compute the ratio
		ratio = torch.exp(current_log_probs - old_log_probs)
		# compute the clipped ratio
		clip_ratio = torch.clamp(ratio, 1 - self.cfg.ppo_clip, 1 + self.cfg.ppo_clip)
		# compute the loss
		policy_loss = -torch.min(ratio * rollout_advantages, clip_ratio * rollout_advantages).sum() / resp_attention_mask.sum()

		# compute the value loss
		value_loss = torch.tensor(0.0)
		if not self.cfg.adv_use_returns:
			values = values * resp_attention_mask
			value_loss = torch.nn.functional.mse_loss(values, rollout_returns)

		# compute the action value loss
		action_value_loss = torch.tensor(0.0)
		if self.cfg.lm_head_model_action_value:
			action_values = action_values * resp_attention_mask
			values_t1 = torch.concat([values[..., 1:], torch.zeros_like(values[..., -1:])], dim=-1)
			rollout_action_values = rollout_rewards + self.cfg.rl_gamma * values_t1.detach()
			action_value_loss = torch.nn.functional.mse_loss(action_values, rollout_action_values)

		kl_loss = torch.tensor(0.0)
		if self.cfg.add_kl_divergence and self.cfg.kl_loss_coeff > 0.0:
			if self.cfg.precompute_kl_logits:
				kl_logit = kl_logit.to(self.cfg.device)
			else:
				with torch.no_grad():
					kl_logit, _, _ = self.kl_model.resp_forward(input_ids=resp_input_ids,	
													attention_mask=attention_mask,
													encoder_outputs=None,
													lm_labels=resp_label_ids,
													return_dict=False)
			kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
			kl_loss = kl_loss_fn(torch.log_softmax(lm_logits, dim=-1), torch.softmax(kl_logit, dim=-1))
			

		total_loss = policy_loss + self.cfg.kl_loss_coeff * kl_loss + self.cfg.val_loss_coeff * value_loss + self.cfg.action_val_loss_coeff * action_value_loss
		average_returns = rollout_returns.sum() / resp_attention_mask.sum()
		average_rewards = rollout_rewards.sum() / resp_attention_mask.sum()
		num_clamped = (ratio != clip_ratio).sum() / ratio.numel()
		step_output = {
			"value_loss": value_loss.item(),
			"action_value_loss": action_value_loss.item(),
			"policy_loss": policy_loss.item(),
			"total_loss": total_loss.item(),
			"average_returns": average_returns.item(),
			"average_rewards": average_rewards.item(),
			"num_clamped": num_clamped.item(),  # reporter will accumulate until log_frequency
			"kl_loss": kl_loss.item(),
		}
		return total_loss, step_output

	@deprecated(reason="can be mostly replaced by train_alternate_k_steps, and num_per_sample is not used")
	def train_alterante_epoch(self, train_batches, num_steps_per_epoch, optimizer, scheduler):
		if self.cfg.use_sl:
			train_iterator = self.iterator.get_data_iterator(train_batches, 'e2e', False, True, -1, resp_use_bspn=self.cfg.resp_use_bspn)
			for step, batch in enumerate(train_iterator):
				inputs, labels = batch
				_, belief_labels, _ = labels

				loss, step_outputs = self.lm_step_fn(inputs, *labels)

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
					if self.reporter is not None:
						self.reporter.step(step_outputs, lr=lr)
			# residual gradient
			if (step + 1) % self.cfg.grad_accum_steps != 0:
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()

				lr = scheduler.get_last_lr()[0]
				if self.reporter is not None:
					self.reporter.step(step_outputs, lr=lr)
		# train PPO
		if self.cfg.use_ppo:
			train_iterator = self.iterator.get_data_iterator(train_batches, 'e2e', False, True, -1, resp_use_bspn=self.cfg.resp_use_bspn)
			rollout_buffer = self.collect_rollout(train_iterator, num_steps_per_epoch)

			for _ in range(self.cfg.ppo_epoch):
				train_iterator = self.iterator.get_data_iterator(train_batches, 'e2e', False, True, -1, resp_use_bspn=self.cfg.resp_use_bspn)
				for step, (batch, *rollout) in enumerate(zip(train_iterator, *rollout_buffer)):
					inputs, labels = batch
					_, _, resp_labels = labels
					
					loss, step_outputs = self.ppo_step_fn(inputs, resp_labels, *rollout)

					if self.cfg.grad_accum_steps > 1:
						loss = loss / self.cfg.grad_accum_steps
					loss.backward()

					torch.nn.utils.clip_grad_norm_(
						self.model.parameters(), self.cfg.max_grad_norm)

					if (step+1) % self.cfg.grad_accum_steps == 0:
						optimizer.step()
						scheduler.step()
						optimizer.zero_grad()

						lr = scheduler.get_last_lr()[0]
						if self.rl_reporter is not None:
							self.rl_reporter.step(step_outputs, lr=lr)
				# residual gradient
				if (step+1) % self.cfg.grad_accum_steps != 0:
					optimizer.step()
					scheduler.step()
					optimizer.zero_grad()

					lr = scheduler.get_last_lr()[0]
					if self.rl_reporter is not None:
						self.rl_reporter.step(step_outputs, lr=lr)
		return

	@deprecated(reason="can be mostly replaced by train_alternate_k_steps")
	def train_alternate_step(self, train_batches, num_steps_per_epoch, optimizer, scheduler, kl_logits):
		train_iterator = self.iterator.get_data_iterator(train_batches, 'e2e', False, True, -1, resp_use_bspn=self.cfg.resp_use_bspn)
		rollout_buffer = self.collect_rollout(train_iterator, num_steps_per_epoch, self.cfg.num_per_sample)

		train_iterator = self.iterator.get_data_iterator(train_batches, 'e2e', False, True, -1, resp_use_bspn=self.cfg.resp_use_bspn)
		for step, (batch, *rollout) in enumerate(zip(train_iterator, *rollout_buffer)):
			inputs, labels = batch
			_, belief_labels, resp_labels = labels

			# train lm
			if self.cfg.use_sl:
				loss, lm_step_outputs = self.lm_step_fn(inputs, *labels)

				if self.cfg.grad_accum_steps > 1:
					loss = loss / self.cfg.grad_accum_steps
				loss.backward()
				torch.nn.utils.clip_grad_norm_(
					self.model.parameters(), self.cfg.max_grad_norm)

				if (step + 1) % self.cfg.grad_accum_steps == 0:
					if self.reporter is not None:
						self.reporter.step(lm_step_outputs)
			
			# train PPO
			if self.cfg.use_ppo:
				for _ in range(self.cfg.ppo_epoch):
					for _step, i_th_rollout in enumerate(zip(*rollout)):
						# each rollout is a list of samples from current @batch, hence the step
						kl_logit = []
						if self.cfg.add_kl_divergence and self.cfg.precompute_kl_logits:
							# kl_logit = kl_logits[step]
							kl_logit = self.db.search('kl_logits', step)
							kl_logit = torch.tensor(kl_logit)
						loss, rl_step_outputs = self.ppo_step_fn(inputs, resp_labels, *i_th_rollout, kl_logit=kl_logit)

						rl_loss, rl_step_outputs = self.ppo_step_fn(inputs, resp_labels, *i_th_rollout)

					if self.cfg.grad_accum_steps > 1:
						rl_loss = rl_loss / self.cfg.grad_accum_steps
					rl_loss.backward()
					torch.nn.utils.clip_grad_norm_(
						self.model.parameters(), self.cfg.max_grad_norm)

					if (step + 1) % self.cfg.grad_accum_steps == 0:
						optimizer.step()
						scheduler.step()
						optimizer.zero_grad()

						lr = scheduler.get_last_lr()[0]
						if self.rl_reporter is not None:
							self.rl_reporter.step(rl_step_outputs, lr=lr)
		# residual gradient
		if (step + 1) % self.cfg.grad_accum_steps != 0:
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()

			lr = scheduler.get_last_lr()[0]
			if self.cfg.use_sl and self.reporter is not None:
				self.reporter.step(lm_step_outputs)
			if self.rl_reporter is not None:
				self.rl_reporter.step(rl_step_outputs, lr=lr)
		return

	def train_alternate_k_steps(self, train_batches, num_steps_per_epoch, optimizer, scheduler, k, kl_logits):
		if not self.cfg.use_sl:
			logger.warning(f"{self.cfg.use_sl=} and {self.cfg.use_ppo=}")

		train_iterator = self.iterator.get_data_iterator(train_batches, 'e2e', False, True, -1, resp_use_bspn=self.cfg.resp_use_bspn)
		rollout_train_iterator = self.iterator.get_data_iterator(train_batches, 'e2e', False, True, -1, resp_use_bspn=self.cfg.resp_use_bspn)

		trained_batches = []
		for step, batch in enumerate(train_iterator):
			if self.cfg.use_ppo:
				trained_batches.append(batch)
			inputs, labels = batch
			_, belief_labels, _ = labels

			if self.cfg.use_sl:
				loss, lm_step_outputs = self.lm_step_fn(inputs, *labels)

				if self.cfg.grad_accum_steps > 1:
					loss = loss / self.cfg.grad_accum_steps
				loss.backward()

				torch.nn.utils.clip_grad_norm_(
					self.model.parameters(), self.cfg.max_grad_norm)

				if (step + 1) % self.cfg.grad_accum_steps == 0:
					optimizer.step()
					optimizer.zero_grad()

					lr = scheduler.get_last_lr()[0]
					if self.reporter is not None:
						self.reporter.step(lm_step_outputs, lr=lr)
			
			if (step + 1) % self.cfg.grad_accum_steps == 0:
				scheduler.step()
			
			# PPO training after k steps
			if self.cfg.use_ppo:
				if (step + 1) % k == 0:
					rollout_buffer = self.collect_rollout(rollout_train_iterator, k, self.cfg.num_per_sample)

					for _ in range(self.cfg.ppo_epoch):
						for _step, (batch, *rollout) in enumerate(zip(trained_batches, *rollout_buffer)):
							inputs, labels = batch
							_, _, resp_labels = labels
							for __step, i_th_rollout in enumerate(zip(*rollout)):
								# each rollout is a list of samples from current @batch, hence the step
								real_kl_i = (step // k) * k + _step
								kl_logit = []
								if self.cfg.add_kl_divergence and self.cfg.precompute_kl_logits:
									# kl_logit = kl_logits[real_kl_i]
									kl_logit = self.db.search('kl_logits', real_kl_i)
									kl_logit = torch.tensor(kl_logit)
								loss, rl_step_outputs = self.ppo_step_fn(inputs, resp_labels, *i_th_rollout, kl_logit=kl_logit)

								if self.cfg.grad_accum_steps > 1:
									loss = loss / self.cfg.grad_accum_steps
								loss.backward()

								torch.nn.utils.clip_grad_norm_(
									self.model.parameters(), self.cfg.max_grad_norm)

								real_step = step * self.cfg.num_per_sample + __step
								if (real_step+1) % self.cfg.grad_accum_steps == 0:
									optimizer.step()
									# scheduler.step() # otherwise need to recalibrate lr
									optimizer.zero_grad()

									lr = scheduler.get_last_lr()[0]
									if self.rl_reporter is not None:
										self.rl_reporter.step(rl_step_outputs, lr=lr)
					# reset trained_batches
					trained_batches = []
		# residual gradient
		if (step + 1) % self.cfg.grad_accum_steps != 0:
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()

			lr = scheduler.get_last_lr()[0]
			if self.cfg.use_sl and self.reporter is not None:
				self.reporter.step(lm_step_outputs)
			if self.rl_reporter is not None:
				self.rl_reporter.step(rl_step_outputs, lr=lr)
		return

	def __update_hyperparams(self, epoch):
		# init
		if len(self.__cfg_backups) == 0:
			self.__cfg_backups['sample_temperature_decay'] = self.cfg.sample_temperature_decay

		if self.cfg.sample_temperature_decay != 1.0:
			self.cfg.sample_temperature = max(self.cfg.sample_temperature * self.cfg.sample_temperature_decay, 1e-3)
		if epoch == self.cfg.epochs:  # done
			self.cfg.sample_temperature = self.__cfg_backups['sample_temperature_decay']
		logger.info('sample_temperature: {}'.format(self.cfg.sample_temperature))
		return

	def precompute_if_needed(self, train_batches, num_steps_per_epoch):
		# check if exist
		if self.db is None:
			return
		if self.db.check_if_db_exists(create=True):
			return
		# precompute
		with open(self.db.tables['kl_logits'], 'wb') as f:
			kl_train_iterator = self.iterator.get_data_iterator(train_batches, 'e2e', False, True, -1, resp_use_bspn=self.cfg.resp_use_bspn)
			for i, batch in tqdm(enumerate(kl_train_iterator), total=num_steps_per_epoch, desc='collecting kl_logits'):
				resp_input_ids, labels = batch
				_, _, resp_label_ids = labels

				if self.cfg.add_gold_demonstrations:
					# collected data will have batch_size*2 as half of them is gold
					resp_input_ids = torch.cat([resp_input_ids, resp_input_ids], dim=0)
					resp_label_ids = torch.cat([resp_label_ids, resp_label_ids], dim=0)
				# convert to tensor
				resp_input_ids = resp_input_ids.to(self.cfg.device)
				resp_label_ids = resp_label_ids.to(self.cfg.device)

				with torch.no_grad():
					attention_mask = torch.where(resp_input_ids == self.reader.pad_token_id, 0, 1)
					kl_logit, _, _ = self.kl_model.resp_forward(input_ids=resp_input_ids,	
													attention_mask=attention_mask,
													encoder_outputs=None,
													lm_labels=resp_label_ids,
													return_dict=False)
				kl_logit = kl_logit.cpu().numpy()
				np.save(f, kl_logit)
		return

	def train_epoch(self, epoch, train_batches, num_steps_per_epoch, optimizer, scheduler, reporter=None):
		self.model.train()
		self.model.zero_grad()

		self.precompute_if_needed(train_batches, num_steps_per_epoch)

		# genearl idea:
		# 1. collect exerpience from current policy
		# 2. train for k epochs on the collected experience
		for l in range(self.cfg.rollout_train_epochs):
			logger.info(f"training rollouts for {l+1}/{self.cfg.rollout_train_epochs} epochs")
			k = self.cfg.alternate_step_k
			if isinstance(k, float):
				k = int(k * num_steps_per_epoch)
			logger.info(f"alteranting LM and PPO every {k=} steps")
			self.train_alternate_k_steps(train_batches, num_steps_per_epoch, optimizer, scheduler, k, [])
		# 3. update old policy network
		update_freq = self.cfg.update_old_policy_interval - 1  # for backward compatibility so self.cfg.update_old_policy_interval=1 means realtime
		if (update_freq > 0	and epoch % update_freq == 0):
			self.model.update_old_policy()
			logger.info("updated old policy network")
		# 4. update any additional hyperparameters
		self.__update_hyperparams(epoch)
		return

	def __compute_num_training_steps(self, num_training_steps_per_epoch):
		if not self.cfg.use_ppo:  # baseline
			return num_training_steps_per_epoch
		if not self.cfg.use_sl:  # only PPO
			return num_training_steps_per_epoch * self.cfg.ppo_epoch
		return num_training_steps_per_epoch  # we do not step PPO during train_alternate_k_steps
	
	def train(self):
		seed_everything(self.cfg.seed, deterministic=self.cfg.deterministic)
		batch_size = self.cfg.batch_size
		if self.cfg.add_gold_demonstrations:
			batch_size = batch_size // 2  # half for gold, half for generated
		train_batches, num_training_steps_per_epoch, _, _ = self.iterator.get_batches(
			"train", batch_size, self.cfg.num_gpus, shuffle=True, seed=self.cfg.seed,
			num_dialogs=self.cfg.num_train_dialogs, excluded_domains=None)

		# for debugging, pick a small sample only
		if self.cfg.debug or self.cfg.pilot_run:
			multiplier = 2 if batch_size < 8 else 1
			cut_off = 20 if self.cfg.debug else 100
			cut_off *= multiplier
			train_batches = train_batches[:cut_off]
			num_training_steps_per_epoch = 0
			train_iterator = self.iterator.get_data_iterator(train_batches, 'e2e', False, True, -1, resp_use_bspn=self.cfg.resp_use_bspn)
			for step in train_iterator:
				num_training_steps_per_epoch += 1
		self._train_batches = copy.deepcopy(train_batches)

		scheduler_num_training_steps_per_epoch = self.__compute_num_training_steps(num_training_steps_per_epoch)
		optimizer, scheduler = self.model.get_optimizer_and_scheduler(scheduler_num_training_steps_per_epoch)

		for epoch in range(1, self.cfg.epochs + 1):
			self._epoch = epoch
			self.train_epoch(epoch, train_batches, num_training_steps_per_epoch, optimizer, scheduler, self.reporter)

			logger.info("done {}/{} epoch".format(epoch, self.cfg.epochs))

			val_output = None
			if not self.cfg.no_validation:
				val_output = self.validation(self.reporter.global_step, epoch)
				if self.ray_reporter is not None:
					self.ray_reporter.report(val_output)
			
			if self.cfg.save_model:
				self.save_model(epoch, val_output=val_output)
		
		# done training, test using the best or last model
		if self.cfg.save_model:
			self.load_best_model()
		if not self.cfg.no_predict:
			logger.info("Testing on test set")
			self.predict()
		return

	def load_best_model(self):
		best_model_path = self._best_performances['mapping'][0][1]
		best_model_path = os.path.join(self.cfg.model_dir, best_model_path)
		logger.info(f"loading best model from {best_model_path}, amonst {self._best_performances['mapping']}")
		self.model.load_model_ckpt(best_model_path)
		return