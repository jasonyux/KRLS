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

from data_utils.mttod.utils import definitions
from trainers.mttod_trainer import MTTODTrainer
from transformers import TopPLogitsWarper, TemperatureLogitsWarper
from torch.distributions import Categorical
from torchmetrics.functional import bleu_score
from transformers.modeling_outputs import BaseModelOutput
from bert_score import BERTScorer
from utils.reporter import WandbReporter, SimpleReporter
from utils.utils import seed_everything, cfg_to_dict, get_or_create_logger, get_mean_std_with_padding

from trainers.ppotod_trainer import PPOLMTrainer


logger = get_or_create_logger(__name__)


class PPORealRLTrainer(PPOLMTrainer):
	def __init__(self, dataset, cfg, model:PPOTODModel, group_name, run_name, test_dataset=None, ray_reporter=None, kl_model=None):
		super(PPORealRLTrainer, self).__init__(dataset, cfg, model, group_name, run_name, test_dataset, ray_reporter, kl_model)
		assert(self.cfg.use_sl == False)
		assert(self.cfg.reward == 'zeros')
		assert(self.cfg.terminal_reward_fn == 'all')
		return

	def compute_terminal_reward(self, resp_pred, resp_label, reward):
		bos_resp_token_id = self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
		eos_resp_token_id = self.reader.get_token_id(definitions.EOS_RESP_TOKEN)
		start_id = np.where(resp_label == bos_resp_token_id)[1]
		end_id = np.where(resp_label == eos_resp_token_id)[1]
		pred_start_id = np.where(resp_pred == bos_resp_token_id)[1]
		pred_end_id = np.where(resp_pred == eos_resp_token_id)[1]
		for i, (sid, eid, psid, peid) in enumerate(zip(start_id, end_id, pred_start_id, pred_end_id)):
			pred_sentence = resp_pred[i, psid+1: peid]
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
			reward[i, peid] += self.cfg.rl_gamma * self.cfg.terminal_reward_scale * terminal_reward
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
		resp_attention_mask = self.get_resp_attention_mask(greedy_preds).cpu().numpy()
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

	def sample_action(self, encoder_outputs, attention_mask, resp_prompt, num_per_sample: int = 1) -> List[torch.Tensor]:
		with torch.no_grad():
			# generate response
			resp_outputs = self.model.generate(
										encoder_outputs=encoder_outputs,
										attention_mask=attention_mask,
										decoder_input_ids=resp_prompt,
										pad_token_id=self.reader.pad_token_id,
										eos_token_id=self.reader.eos_token_id,
										max_new_tokens=100,
										do_sample=True,
										num_beams=self.cfg.beam_size,
										early_stopping=False,
										temperature=1.1,
										top_k=self.cfg.top_k,
										top_p=0.9,
										repetition_penalty=self.cfg.repetition_penalty,
										num_return_sequences=num_per_sample,
										decoder_type="resp")
		regrouped_resp_outputs = []
		for i in range(num_per_sample):
			regrouped_resp_outputs.append(resp_outputs[i::num_per_sample])
		return regrouped_resp_outputs

	def prepare_decoder_resp_input(self, resp_label_ids):
		# find the bos token in resp_label_ids
		# and pad form the left to fill in spaces
		resp_label_ids = resp_label_ids.cpu()
		bos_resp_token_id = self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
		pad_token_id = self.reader.pad_token_id

		resp_prompts = []
		for i in range(resp_label_ids.shape[0]):
			bos_positions = torch.where(resp_label_ids[i] == bos_resp_token_id)[0]
			bos_position = bos_positions[0]
			resp_prompts.append(resp_label_ids[i][:bos_position+1])
		# pad the prompts
		max_len = max([len(prompt) for prompt in resp_prompts])
		for i in range(len(resp_prompts)):
			padded_prompt = [pad_token_id] * (max_len - len(resp_prompts[i])) + resp_prompts[i].tolist()
			resp_prompts[i] = torch.tensor(padded_prompt, dtype=torch.long)
		resp_prompts = torch.stack(resp_prompts, dim=0)
		return resp_prompts


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
				encoder_outputs = self.model(input_ids=inputs,
												attention_mask=attention_mask,
												return_dict=False,
												encoder_only=True,
												add_auxiliary_task=False)

				span_outputs, encoder_hidden_states = encoder_outputs

				if isinstance(encoder_hidden_states, tuple):
					last_hidden_state = encoder_hidden_states[0]
				else:
					last_hidden_state = encoder_hidden_states

				# wrap up encoder outputs
				encoder_outputs = BaseModelOutput(
					last_hidden_state=last_hidden_state)


				resp_prompt = self.prepare_decoder_resp_input(resp_labels)
				resp_prompt = resp_prompt.to(self.cfg.device)

			lm_preds = self.sample_action(encoder_outputs, attention_mask, resp_prompt, self.cfg.num_per_sample)
			lm_preds_probs = []
			rewards = []
			advantages = []
			returns = []
			for lm_pred in lm_preds:
				with torch.no_grad():
					lm_logits, values, action_values = self.model.resp_forward(input_ids=inputs,	
																attention_mask=attention_mask,
																encoder_outputs=None,
																lm_labels=lm_pred.to(inputs.device),
																use_old_model=True,
																return_dict=False)
				
				# collect the token ids
				lm_probs = torch.softmax(lm_logits, dim=-1) # [batch_size, prob_of_each_word]

				# collect the probabilities
				lm_preds_prob = torch.gather(lm_probs, dim=-1, index=lm_pred.unsqueeze(-1)).squeeze(-1)
				lm_preds_probs.append(lm_preds_prob.detach().cpu().numpy())
				
				# print(self.reader.tokenizer.batch_decode(lm_pred))
				reward, advantage, return_ = self.compute_advantage_n_returns(lm_pred, resp_labels, values, step)
				rewards.append(reward)
				advantages.append(advantage)
				returns.append(return_)

			collected_token_ids.append([lm_pred.detach().cpu().numpy() for lm_pred in lm_preds])
			collected_probs.append(lm_preds_probs)
			collected_rewards.append(rewards)
			collected_advantages.append(advantages)
			collected_returns.append(returns)
			
			pbar.update(1)
			if step + 1 == num_steps_per_epoch:
				break
		pbar.close()
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

		attention_mask = torch.where(resp_input_ids == self.reader.pad_token_id, 0, 1)
		lm_logits, values, action_values = self.model.resp_forward(input_ids=resp_input_ids,	
												attention_mask=attention_mask,
												encoder_outputs=None,
												lm_labels=rollout_token_ids,
												return_dict=False)

		# resp_attention_mask = torch.where(resp_label_ids == self.reader.pad_token_id, 0, 1)
		resp_attention_mask = self.get_resp_attention_mask(rollout_token_ids)
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
													lm_labels=rollout_token_ids,
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