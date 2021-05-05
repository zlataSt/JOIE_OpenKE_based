import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class TransE(Model):

	def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
		super(TransE, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		# print('in TransE init ent tot rel tot and dim', self.ent_tot, self.rel_tot, self.dim)
		# print('in TransE  shape of self.ent_embeddings.weight.data', self.ent_embeddings.weight.data.shape)
		print('init self.ent_embeddings.weight.data[0][0]',
			  self.ent_embeddings.weight.data[0][0])
		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
			print('init self.ent_embeddings.weight.data[0][0]',
				  self.ent_embeddings.weight.data[0][0])
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False


	def _calc(self, h, t, r, mode):
		# print('Transe in _cala h', h.shape)
		# print('Transe in _cala t', t.shape)
		# print('Transe in _cala r', r.shape)
		print('calc self.ent_embeddings.weight.data[0][0]',
			  self.ent_embeddings.weight.data[0][0])
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		# print('Transe in _cala h', h.shape)
		# print('Transe in _cala t', t.shape)
		# print('Transe in _cala r', r.shape)
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		# print('score',score.shape)
		score = torch.norm(score, self.p_norm, -1).flatten()
		# print('score in _calc', score, score.shape)
		return score

	def forward(self, data):
		# print('Transe in forward data', data, len(data))
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		# print('Transe in forward batch_h', batch_h, len(batch_h), batch_h.shape)
		# print('Transe in forward batch_t', batch_t, len(batch_t), batch_t.shape)
		# print('Transe in forward batch_r', batch_r, len(batch_r), batch_r.shape)
		# print('Transe in forward mode', mode, len(mode))
		print('forward self.ent_embeddings.weight.data[0][0]', self.ent_embeddings.weight.data[0][0])
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		# print('Transe in forward h', h, len(h), h.shape)
		# print('Transe in forward t', t, len(t), t.shape)
		# print('Transe in forward r', r, len(r), r.shape)
		score = self._calc(h ,t, r, mode)
		# print('Transe in score r', score, len(score), score.shape)
		print('forward self.ent_embeddings.weight.data[0][0]',
			  self.ent_embeddings.weight.data[0][0])
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def regularization(self, data):
		# print('TransE regularization data', data, len(data), data.shape)
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		print('regul self.ent_embeddings.weight.data[0][0]',
			  self.ent_embeddings.weight.data[0][0])
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		# print('TransE h', h, len(h), h.shape)
		# print('TransE t', t, len(t), t.shape)
		# print('TransE r', r, len(r), r.shape)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul

	def predict(self, data):
		# print('TransE predict data', data, len(data))
		score = self.forward(data)
		print('predict self.ent_embeddings.weight.data[0][0]',
			  self.ent_embeddings.weight.data[0][0])
		# print('TransE predict score', score, len(score), score.shape)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()