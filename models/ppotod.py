import torch
import numpy as np
import copy

from torch import nn
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import get_linear_schedule_with_warmup, get_constant_schedule

from models.mttod import ModelBase, T5WithTokenSpan
from data_utils.mttod.utils import definitions
from utils.utils import get_or_create_logger
from utils.globals import _PPOTOD_CONFIG


logger = get_or_create_logger(__name__)


class T5WithResponsePPO(T5WithTokenSpan):
	def __init__(self, config, num_span, special_token_ids, extra_cfg: _PPOTOD_CONFIG, da_token_ids=[]):
		super(T5WithResponsePPO, self).__init__(config, num_span)
		self._special_token_ids = special_token_ids
		self._da_token_ids = da_token_ids
		self.cfg = extra_cfg
		# not used if shared_value_head is True
		self.value_function = torch.nn.Sequential(
			torch.nn.Linear(self.config.d_model, self.config.d_model),
			torch.nn.ReLU(),
			torch.nn.Linear(self.config.d_model, 1)
		)
		# TODO: not used
		self.action_value_function = type(self.lm_head)(
			self.config.d_model, self.config.vocab_size, bias=False)
	
	def initialize_additional_decoder(self, cfg: _PPOTOD_CONFIG):
		decoder_config = copy.deepcopy(self.config)
		decoder_config.is_decoder = True
		decoder_config.is_encoder_decoder = False

		self.resp_decoder = type(self.decoder)(decoder_config, self.shared)
		self.resp_lm_head = type(self.lm_head)(
			self.config.d_model, self.config.vocab_size, bias=False)
		self.action_value_function = type(self.lm_head)(
			self.config.d_model, self.config.vocab_size, bias=False)

		self.resp_decoder.load_state_dict(self.decoder.state_dict())
		if cfg.lm_head_init: # if separate value head, at least load the lm weights
			self.resp_lm_head.load_state_dict(self.lm_head.state_dict())
		if cfg.action_value_lm_head_init:
			self.action_value_function.load_state_dict(self.lm_head.state_dict())
		return

	def compute_scaled_lm_loss(self, lm_logits, lm_labels):
		if self.cfg.lm_scale == "none":
			lm_loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
			lm_loss = lm_loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
			return lm_loss  # normal CE loss
		
		lm_logits = lm_logits.permute(0, 2, 1)  # CE loss requires this shape

		resp_pred = lm_logits.argmax(dim=1).detach().cpu().numpy()
		resp_label = lm_labels.detach().cpu().numpy()
		lm_loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction='none')
		special_token_mask = np.isin(resp_label, self._special_token_ids)
		if self.cfg.lm_scale == "sentence_error":
			sent_error = 1.0 + np.sum(special_token_mask * (resp_pred != resp_label), axis=1, keepdims=True)  # if all correct, just cross entropy
			sent_error = sent_error * np.ones_like(resp_label)
			sent_error = torch.from_numpy(sent_error).to(lm_logits.device)
			lm_loss = lm_loss_fct(lm_logits, lm_labels)
			return (lm_loss * sent_error).mean()
		elif self.cfg.lm_scale == "token_error":
			special_token_scale = self.cfg.special_token_error_scale - 1.0
			token_error = np.ones_like(resp_label).astype(float)  # at least non-zero otherwise loss is zero
			token_error += (resp_label != resp_pred).astype(float)  # add one if a token is wrong
			token_error += special_token_scale * special_token_mask * (resp_label != resp_pred)  # add more if a special token is wrong
			token_error = torch.from_numpy(token_error).to(lm_logits.device)
			lm_loss = lm_loss_fct(lm_logits, lm_labels)
			return (lm_loss * token_error).mean()
		raise NotImplementedError  # shouldn't reach here

	def forward(self,
				input_ids=None,
				attention_mask=None,
				decoder_input_ids=None,
				encoder_outputs=None,
				past_key_values=None,
				inputs_embeds=None,
				decoder_inputs_embeds=None,
				span_labels=None,
				lm_labels=None,
				use_cache=None,
				output_attentions=None,
				output_hidden_states=None,
				return_dict=None,
				encoder_only=None,
				add_auxiliary_task=None,
				decoder_type=None):
		"""forward has to be here or its base class as `generate` is called from here
		"""
		use_cache = use_cache if use_cache is not None else self.config.use_cache
		return_dict = return_dict if return_dict is not None else self.config.return_dict

		span_loss, pred_spans, span_logits = 0, None, None

		if encoder_outputs is None:
			encoder_outputs = self.encoder(input_ids=input_ids,
														attention_mask=attention_mask,
														inputs_embeds=inputs_embeds,
														return_dict=return_dict)

			if return_dict:
				encoder_hidden_states = encoder_outputs.last_hidden_state
			else:
				encoder_hidden_states = encoder_outputs[0]

			# encoder forward to obtain last hidden state for each token
			hs = encoder_hidden_states * (self.model_dim ** -0.5)

			if add_auxiliary_task:
				# loss for span prediction, which is encoder_hidden_state + linear head
				span_loss, pred_spans, span_logits = self.predict_span(
					hs, attention_mask, span_labels)

		else:
			if isinstance(encoder_outputs, tuple):
				encoder_hidden_states = encoder_outputs[0]
			else:
				encoder_hidden_states = encoder_outputs.last_hidden_state

		if encoder_only:
			return (span_loss, pred_spans, span_logits), encoder_outputs

		if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
			decoder_input_ids = self._shift_right(lm_labels) # input starts from <sos>, hence shift right

		if decoder_type == "resp":
			decoder = self.resp_decoder # an additional decoder for response
			lm_head = self.resp_lm_head
		else:
			decoder = self.decoder # the original decoder and lm head for T5
			lm_head = self.lm_head

		if past_key_values is not None:
			assert lm_labels is None, "Decoder should not use cached key value states when training"
			if decoder_input_ids is not None:
				decoder_input_ids = decoder_input_ids[:, -1:]
			if decoder_inputs_embeds is not None:
				decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

		decoder_outputs = decoder(input_ids=decoder_input_ids,
								  inputs_embeds=decoder_inputs_embeds,
								  past_key_values=past_key_values,
								  encoder_hidden_states=encoder_hidden_states,
								  encoder_attention_mask=attention_mask,
								  use_cache=use_cache,
								  return_dict=return_dict)

		sequence_output = decoder_outputs[0]

		sequence_output = sequence_output * (self.model_dim ** -0.5)

		lm_logits = lm_head(sequence_output) # final prediction = probability of each token

		lm_loss = None
		if lm_labels is not None:
			if decoder_type == "resp":  # baseline idea: scale per token loss by error
				lm_loss = self.compute_scaled_lm_loss(lm_logits, lm_labels)
			else:  # TODO: hierarchical loss
				lm_loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
				lm_loss = lm_loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

		# for training
		if not return_dict:
			pred_lm = torch.argmax(lm_logits, dim=-1)
			outputs = (lm_loss, pred_lm,) + \
				(span_loss, pred_spans, span_logits, encoder_hidden_states) + \
					decoder_outputs[1:]

		# for prediction
		else:
			outputs = Seq2SeqLMOutput(
				loss=lm_loss,
				logits=lm_logits,
				past_key_values=decoder_outputs.past_key_values,
				decoder_hidden_states=decoder_outputs.hidden_states,
				decoder_attentions=decoder_outputs.attentions,
				cross_attentions=decoder_outputs.cross_attentions,
				encoder_last_hidden_state=encoder_outputs.last_hidden_state,
				encoder_hidden_states=encoder_outputs[1] if len(
					encoder_outputs) > 1 else None,
				encoder_attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
		return outputs


class PPOTODModel(ModelBase):
	def __init__(self, config: _PPOTOD_CONFIG, reader_vocab_size, tokenizer):
		super(PPOTODModel, self).__init__(config)

		if self.cfg.ckpt is not None:
			model_path = self.cfg.ckpt
			initialize_additional_decoder = False
		elif self.cfg.train_from is not None:
			model_path = self.cfg.train_from
			initialize_additional_decoder = False
		else:
			model_path = self.cfg.backbone
			initialize_additional_decoder = True

		self._num_span = len(definitions.EXTRACTIVE_SLOT)
		# if self.cfg.use_all_sp_tokens:
		# 	special_token_ids = [v for k, v in tokenizer.get_added_vocab().items() if k.startswith('[') and not k.startswith('[db') and not k.startswith('[PAD]')]
		# else:
		# 	special_token_ids = [v for k, v in tokenizer.get_added_vocab().items() if k.startswith('[value')]
		da_token_ids = []
		resp_special_token_ids = []
		for k, v in tokenizer.get_added_vocab().items():
			if k.startswith('[') and not k.startswith('[db') and not k.startswith('[PAD]'):
				if k.startswith('[value'):
					resp_special_token_ids.append(v)
				else:
					da_token_ids.append(v)
		
		self.model = T5WithResponsePPO.from_pretrained(
			model_path, 
			num_span=self._num_span,
			special_token_ids=resp_special_token_ids,
			da_token_ids=da_token_ids,
			extra_cfg=self.cfg
		)
		# align token ids, needed since GODEL uses a different pad_token_id than t5
		self.model.config.pad_token_id = tokenizer.pad_token_id
		self.model.config.eos_token_id = tokenizer.eos_token_id
		self.model.config.decoder_start_token_id = tokenizer.pad_token_id

		self._reader_vocab_size = reader_vocab_size
		self.model.resize_token_embeddings(reader_vocab_size)
		if initialize_additional_decoder:
			self.model.initialize_additional_decoder(self.cfg)
		self.adjust_dropout(self.cfg.dropout)
		
		self.model.to(self.cfg.device)
		if config.update_old_policy_interval > 1:  # need wait
			logger.info("Initializing old policy")
			self.old_model = copy.deepcopy(self.model)	
			self.old_model.to(self.cfg.device)
		else:
			self.old_model = self.model
		return

	def adjust_dropout(self, dropout):
		self.model.encoder.dropout.p = dropout
		self.model.decoder.dropout.p = dropout
		self.model.resp_decoder.dropout.p = dropout
		return

	def update_old_policy(self):
		self.old_model = copy.deepcopy(self.model)
		return

	def load_model_ckpt(self, ckpt_path):
		config = copy.deepcopy(self.model.config)
		reader_vocab_size = self._reader_vocab_size
		self.model = T5WithResponsePPO.from_pretrained(
			ckpt_path, 
			num_span=self._num_span, 
			special_token_ids=self.model._special_token_ids,
			extra_cfg=self.cfg
		)
		self.model.config = config
		self.model.resize_token_embeddings(reader_vocab_size)
		self.model.to(self.cfg.device)
		return

	def forward(self, *args, **kwargs):
		return self.model(*args, **kwargs)
	
	def resp_forward(self,
				input_ids=None,
				attention_mask=None,
				decoder_input_ids=None,
				encoder_outputs=None,
				past_key_values=None,
				inputs_embeds=None,
				decoder_inputs_embeds=None,
				span_labels=None,
				lm_labels=None,
				output_attentions=None,
				output_hidden_states=None,
				return_dict=None,
				encoder_only=None,
				use_old_model=False,
				return_decoder_outputs=False,
				add_auxiliary_task=None):
		
		return_dict = return_dict if return_dict is not None else self.config.return_dict

		span_loss, pred_spans, span_logits = 0, None, None
		if use_old_model:
			model: T5WithResponsePPO = self.old_model  # used when collecting data
		else:
			model: T5WithResponsePPO = self.model

		if encoder_outputs is None:
			encoder_outputs = model.encoder(input_ids=input_ids,
													attention_mask=attention_mask,
													inputs_embeds=inputs_embeds,
													return_dict=return_dict)

			if return_dict:
				encoder_hidden_states = encoder_outputs.last_hidden_state
			else:
				encoder_hidden_states = encoder_outputs[0]

			# encoder forward to obtain last hidden state for each token
			hs = encoder_hidden_states * (model.model_dim ** -0.5)
		else:
			if isinstance(encoder_outputs, tuple):
				encoder_hidden_states = encoder_outputs[0]
			else:
				encoder_hidden_states = encoder_outputs.last_hidden_state

		if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
			decoder_input_ids = model._shift_right(lm_labels) # input starts from <sos>, hence shift right

		decoder = model.resp_decoder # an additional decoder for response
		lm_head = model.resp_lm_head
		value_head = model.value_function
		
		decoder_outputs = decoder(input_ids=decoder_input_ids,
								  inputs_embeds=decoder_inputs_embeds,
								  past_key_values=past_key_values,
								  encoder_hidden_states=encoder_hidden_states,
								  encoder_attention_mask=attention_mask,
								  return_dict=return_dict)

		sequence_output = decoder_outputs[0]
		sequence_output = sequence_output * (model.model_dim ** -0.5)

		lm_logits = lm_head(sequence_output) # final prediction = logit of each token
		action_value_logits = None
		if self.cfg.lm_head_model_action_value:
			lm_tokens = lm_logits.argmax(dim=-1)
			action_value_logits = lm_logits.gather(-1, lm_tokens.unsqueeze(-1))
			action_value_logits = action_value_logits.view(-1, action_value_logits.size(1))
		else:
			pass
			# TODO: action_value_function
		value_logits = torch.zeros_like(lm_logits)
		if not self.cfg.adv_use_returns:  # if adv use return, we don't need value function
			value_logits = value_head(sequence_output) # value function for current state
			value_logits = value_logits.view(-1, value_logits.size(1))
		if return_decoder_outputs:
			return decoder_outputs, lm_logits, value_logits, action_value_logits
		return lm_logits, value_logits, action_value_logits

	def generate(self, *args, **kwargs):
		return self.model.generate(*args, **kwargs)

	def save_pretrained(self, *args, **kwargs):
		return self.model.save_pretrained(*args, **kwargs)

	def get_optimizer_and_scheduler(self, num_traininig_steps_per_epoch):
		num_train_steps = (num_traininig_steps_per_epoch *
			self.cfg.epochs) // self.cfg.grad_accum_steps

		if self.cfg.warmup_steps >= 0:
			num_warmup_steps = self.cfg.warmup_steps
		else:
			num_warmup_steps = int(num_train_steps * self.cfg.warmup_ratio)

		logger.info(f"Total training steps = {num_train_steps}, {num_warmup_steps=}")

		optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

		if self.cfg.no_learning_rate_decay:
			scheduler = get_constant_schedule(optimizer)
		else:
			scheduler = get_linear_schedule_with_warmup(
				optimizer,
				num_warmup_steps=num_warmup_steps,
				num_training_steps=num_train_steps)

		return optimizer, scheduler