from layer import *
import numpy as np
import torch as pt
import torchvision as ptv
class MLP(pt.nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		self.fc1 = pt.nn.Linear(54, 108)
		self.fc2 = pt.nn.Linear(108, 128)
		self.fc3 = pt.nn.Linear(128, 27)
	def __call__(self, din):
		return self.forward(din)
	def forward(self, din):
		din = din.view(-1, 54)
		dout = pt.nn.functional.sigmoid(self.fc1(din))
		dout = pt.nn.functional.sigmoid(self.fc2(dout))
		dout = pt.nn.functional.sigmoid(self.fc3(dout))
		return dout

