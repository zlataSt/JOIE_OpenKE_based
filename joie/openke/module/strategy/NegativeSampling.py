from .Strategy import Strategy

class NegativeSampling(Strategy):

	def __init__(self, model=None, loss=None, batch_size=256, regul_rate = 0.0, l3_regul_rate = 0.0):
		super(NegativeSampling, self).__init__()
		self.model = model
		self.loss = loss
		self.batch_size = batch_size
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate

	def _get_positive_score(self, score):
		# print('_get_positive_score in NegativeSampling len score and batch_size', len(score), self.batch_size)
		positive_score = score[:self.batch_size]
		positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
		# from [1, 2, 3] to [[1], [2], [3]]
		return positive_score

	def _get_negative_score(self, score):
		negative_score = score[self.batch_size:]
		negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
		return negative_score

	def forward(self, data):
		print('forward negative predict self.ent_embeddings.weight.data[0][0]',
			  self.model.ent_embeddings.weight.data[0][0])
		score = self.model(data)
		# print('data in NegativeSampling', data, type(data), len(data))
		# print('score in NegativeSampling', score, type(score), len(score), score.shape)
		p_score = self._get_positive_score(score)
		n_score = self._get_negative_score(score)
		# print('p_score in NegativeSampling',  type(p_score), len(p_score), p_score.shape)
		# print('n_score in NegativeSampling',  type(n_score), len(n_score), n_score.shape)
		loss_res = self.loss(p_score, n_score)
		# print('loss_res in NegativeSampling', loss_res, type(loss_res), len(loss_res))
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		# print('loss_res in NegativeSampling', loss_res, type(loss_res), len(loss_res))
		return loss_res