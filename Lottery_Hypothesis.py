#source code : https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch/blob/e03ca8ac4178be3a2bff2b82c858bf5e58dd3967/main.py#L269

class Lottery_Hypothesis():
	def __init__(self, net, prune_percent, prune_iter, 
				train_epochs, train_loader, val_loader, test_loader, PATH, SAVE_PATH, learning_rate, loss_fn):
		self.net = net
		self.device = torch.cuda.current_device()
		self.prune_percent = prune_percent
		self.prune_iter = prune_iter
		self.train_epochs = train_epochs

		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader

		self.PATH = PATH #contains weights of initial state
		self.SAVE_PATH = SAVE_PATH

		self.learning_rate = learning_rate
		self.loss_fn = loss_fn

		self.save_init_weights()
		self.init_mask()

	def save_init_weights(self):
		torch.save(self.net.state_dict(), self.PATH)

	def load_init_weights(self):
		self.net.load_state_dict(torch.load(self.PATH))


	def init_mask(self):
		layer = 0
		for name, param in self.net.named_parameters():
			if 'weight' in name:
				layer += 1

		self.mask = [None] * layer
		layer = 0
		for name, param in self.net.named_parameters():
			if 'weight' in name:
				weight = param.data.cpu().numpy()
				self.mask[layer] = np.ones_like(weight)
				layer +=1

	def update_mask(self):
		layer = 0
		for name, param in self.net.named_parameters():
			if 'weight' in name:
				weight = param.data.cpu().numpy()
				alive = weight[np.nonzero(weight)]
				percentile_val = np.percentile(abs(alive), self.prune_percent)
				new_mask = np.where(abs(weight) < percentile_val, 0, self.mask[layer])
				self.mask[layer] = new_mask
				layer+=1

	def prepare_train(self):
		layer = 0
		self.load_init_weights()
		for name, param in self.net.named_parameters():
			if 'weight' in name:
				weight = param.data.cpu().numpy()
				param.data = torch.from_numpy(weight * self.mask[layer]).to(self.device)
				layer += 1


	def train(self):

		EPS = 1e-6


		self.optimizer = optim.Adam(self.net.parameters(), lr = self.learning_rate)

		avg_losses = []
		for epoch in range(self.train_epochs):
			losses = []
			for step, (x, y) in enumerate(train_loader):

				x = x.to(self.device)
				y = y.to(self.device)

				self.optimizer.zero_grad()
				y_ = self.net(x)
				loss = self.loss_fn(y_, y)
				loss.backward()

				for name, param in self.net.named_parameters():
					if 'weight' in name:
						weights = param.data.cpu().numpy()
						grads = param.grad.data.cpu().numpy()
						grads = np.where(weights < EPS, 0, grads)
						param.grad.data = torch.from_numpy(grads).to(self.device)

				self.optimizer.step()
				losses.append(loss.item())

			avg_losses.append(sum(losses) / len(losses))

		return avg_losses

	def test(self):
		pass

	def prune(self):
		for step in trange(self.prune_iter):
			self.prepare_train()
			self.train()
			self.update_mask()

		torch.save(self.net.state_dict(self.SAVE_PATH))