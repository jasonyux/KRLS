import torch
import numpy as np
import os
import json
import copy
import re
import pickle
import ast

from pathlib import Path
from typing import *
from tqdm.auto import tqdm
from models.ppotod import PPOTODModel

from data_utils.mttod.utils import definitions
from trainers.mttod_trainer import MTTODTrainer
from transformers import TopPLogitsWarper, TemperatureLogitsWarper
from torch.distributions import Categorical
from torchmetrics.functional import bleu_score
from bert_score import BERTScorer
from utils.reporter import WandbReporter, SimpleReporter
from utils.utils import seed_everything, cfg_to_dict, get_or_create_logger, get_mean_std_with_padding

from trainers.ppotod_trainer import PPOLMTrainer


logger = get_or_create_logger(__name__)


class PGLMTrainer(PPOLMTrainer):
	def __init__(self, dataset, cfg, model:PPOTODModel, group_name, run_name, test_dataset=None, ray_reporter=None, kl_model=None):
		super(PGLMTrainer, self).__init__(dataset, cfg, model, group_name, run_name, test_dataset, ray_reporter, kl_model)
		return

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

			if self.cfg.rl_algo == "off-policy":
				# collect the probabilities
				lm_preds = [resp_labels for _ in range(num_per_sample)]
				lm_preds_probs = []
				for lm_pred in lm_preds:
					lm_preds_prob = torch.gather(lm_probs, dim=-1, index=lm_pred.unsqueeze(-1)).squeeze(-1)
					lm_preds_probs.append(lm_preds_prob.detach().cpu().numpy())
				collected_probs.append(lm_preds_probs)

				rewards = []
				advantages = []
				returns = []
				for lm_pred in lm_preds:
					reward, advantage, return_ = self.compute_advantage_n_returns(lm_pred, resp_labels, values, step, is_gold=True)
					rewards.append(reward)
					advantages.append(advantage)
					returns.append(return_)
				collected_rewards.append(rewards)
				collected_advantages.append(advantages)
				collected_returns.append(returns)

			elif self.cfg.rl_algo == "ssemi-on-policy":
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


		resp_attention_mask = self.get_resp_attention_mask(resp_label_ids)
		if self.cfg.adv_use_returns:
			rollout_advantages = rollout_returns
		rollout_advantages = rollout_advantages * resp_attention_mask
		rollout_returns = rollout_returns * resp_attention_mask

		# we are not using the value/Q estimates/GAE
		value_loss = torch.tensor(0.0)

		# used for KL divergence, if enabled
		attention_mask = torch.where(resp_input_ids == self.reader.pad_token_id, 0, 1)
		lm_logits, _, _ = self.model.resp_forward(input_ids=resp_input_ids,
													attention_mask=attention_mask,
													encoder_outputs=None,
													lm_labels=resp_label_ids,
													return_dict=False)

		if self.cfg.rl_algo == "off-policy":
			# compute PG: w_t * Q * log(P_theta(gold))
			# compute the log probs
			log_probs = torch.log_softmax(lm_logits, dim=-1)
			log_probs = torch.gather(log_probs, dim=-1, index=resp_label_ids.unsqueeze(-1)).squeeze(-1)
			# weights w_t \approx P_theta(gold) in their case
			with torch.no_grad():
				importance_weights = torch.softmax(lm_logits, dim=-1)
				importance_weights = torch.gather(importance_weights, dim=-1, index=resp_label_ids.unsqueeze(-1)).squeeze(-1)
			
			resp_attention_mask = resp_attention_mask.to(lm_logits.device)
			policy_loss = importance_weights * rollout_advantages * log_probs * resp_attention_mask
			policy_loss = -torch.sum(policy_loss) / resp_attention_mask.sum()
		elif self.cfg.rl_algo == "ssemi-on-policy":  # ours
			# compute PG: Q * log(P_theta(sampled))
			log_probs = torch.log_softmax(lm_logits, dim=-1)
			current_log_probs = torch.gather(log_probs, dim=-1, index=rollout_token_ids.unsqueeze(-1)).squeeze(-1)

			resp_attention_mask = resp_attention_mask.to(current_log_probs.device)
			policy_loss = rollout_advantages * current_log_probs * resp_attention_mask
			policy_loss = -torch.sum(policy_loss) / resp_attention_mask.sum()


		kl_loss = torch.tensor(0.0)
		if self.cfg.add_kl_divergence and self.cfg.kl_loss_coeff > 0.0:
			attention_mask = torch.where(resp_input_ids == self.reader.pad_token_id, 0, 1)
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
			# with torch.no_grad():
			# 	kl_logit, _, _ = self.kl_model.resp_forward(input_ids=resp_input_ids,	
			# 									attention_mask=attention_mask,
			# 									encoder_outputs=None,
			# 									lm_labels=rollout_token_ids,
			# 									return_dict=False)
			# curr_gen_logits, _, _ = self.model.resp_forward(input_ids=resp_input_ids,	
			# 									attention_mask=attention_mask,
			# 									encoder_outputs=None,
			# 									lm_labels=rollout_token_ids,
			# 									return_dict=False)
			# kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
			# # memory issue
			# kl_loss = kl_loss_fn(torch.log_softmax(curr_gen_logits.cpu(), dim=-1), torch.softmax(kl_logit.cpu(), dim=-1))
			# kl_loss = kl_loss.to(policy_loss.device)

		total_loss = policy_loss + self.cfg.kl_loss_coeff * kl_loss + self.cfg.val_loss_coeff * value_loss
		average_returns = rollout_returns.sum() / resp_attention_mask.sum()
		average_rewards = rollout_rewards.sum() / resp_attention_mask.sum()
		step_output = {
			"value_loss": value_loss.item(),
			"action_value_loss": 0.0,
			"policy_loss": policy_loss.item(),
			"total_loss": total_loss.item(),
			"average_returns": average_returns.item(),
			"average_rewards": average_rewards.item(),
			"kl_loss": kl_loss.item(),
		}
		return total_loss, step_output