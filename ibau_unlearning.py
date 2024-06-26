import torch
from tqdm import tqdm
import csv
from test_model import test_epoch
from utils.utils import save_checkpoint_only, progress_bar, normalization

class IBAUUnlearning:
    def __init__(self, model, criterion, arg, epochs=10, lr=0.01):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=arg.schedule, gamma=arg.gamma)
        self.criterion = criterion
        self.args = arg


    def unlearn(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            for data in self.dataloader:
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.calculate_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                
    ### define the inner loss L2
    def loss_inner(perturb, model_params):
        images = inputs[0].to(device)
        labels = labels[0].long().to(device)
	#     per_img = torch.clamp(images+perturb[0],min=0,max=1)
		per_img = images+perturb[0]
		per_logits = model.forward(per_img)
		loss = F.cross_entropy(per_logits, labels, reduction='none')
		loss_regu = torch.mean(-loss) +0.001*torch.pow(torch.norm(perturb[0]),2)
		return loss_regu

	### define the outer loss L1
	def loss_outer(perturb, model_params):
		portion = 0.01
		images, labels = inputs[batchnum].to(device), labels[batchnum].long().to(device)
		patching = torch.zeros_like(images, device='cuda')
		number = images.shape[0]
		rand_idx = random.sample(list(np.arange(number)),int(number*portion))
		patching[rand_idx] = perturb[0]
	#     unlearn_imgs = torch.clamp(images+patching,min=0,max=1)
		unlearn_imgs = images+patching
		logits = model(unlearn_imgs)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(logits, labels)
		return loss

    def calculate_loss(self, outputs, labels):
        # Implement the minimax loss calculation here
        return torch.nn.functional.cross_entropy(outputs, labels)

# Usage
# model = YourModel()
# dataloader = YourDataLoader()
# ibau = IBAU(model, dataloader)
# ibau.unlearn()
