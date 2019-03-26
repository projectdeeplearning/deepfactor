from model import *
from model import MLP
import numpy as np
import csv
import pandas as pd
import torch as pt
import torchvision as ptv
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
	dataname = 'data.csv'
	data = pd.read_csv(dataname)
	data = data.fillna(0)
	data = data.values
	model = MLP()
	data1 = data[0:10]
	data1 = np.array(data1, dtype=np.float64)
	tensor = pt.from_numpy(data1)
	tensor = tensor.type(pt.FloatTensor)
	print model(tensor)

# def train(model, Xtrain, Ytrain, batch_size):
#     model.train()
#     n = len(Xtrain)
#     indexes = np.arange(n)
#     np.random.shuffle(indexes)
#     X = Xtrain[indexes].copy()
#     Y = Ytrain[indexes].copy()
#     acc, tol, train_loss = 0, 0, 0
#     for idx in range(n//batch_size):
#         x = X[idx*batch_size:(idx+1)*batch_size]
#         y = Y[idx*batch_size:(idx+1)*batch_size]
#         img = np.reshape(x, (len(x), 3, 32, 32)) 
#         pred = model(img)
#         loss = loss_fun(pred, y)
#         model.backward(loss_fun.backward())
#         model.step()
#         train_loss += loss
#         tol += y.shape[0]
#         predictions = np.argmax(pred, axis = 1).reshape(1, -1)
#         label = np.argmax(y, axis = 1).reshape(1, -1)
#         acc += np.sum((predictions == label) == 1)
#         np.savetxt('prediction.txt', predictions)
#     train_loss /= (n//batch_size)
#     acc = float(acc)
#     acc /= tol
#     return train_loss, accC

# def val(model, XVal, YVal, batch_size):
#     model.val()
#     n = len(XVal)
#     indexes = np.arange(n)
#     np.random.shuffle(indexes)
#     X = XVal[indexes].copy()
#     Y = YVal[indexes].copy()
#     acc, tol, train_loss = 0, 0, 0
#     for idx in range(n//batch_size):
#         x = X[idx*batch_size:(idx+1)*batch_size]
#         y = Y[idx*batch_size:(idx+1)*batch_size]
#         img = np.reshape(x, (len(x), 3, 32, 32)) 
#         pred = model(img)
#         loss = loss_fun(pred, y)
#         train_loss += loss
#         tol += y.shape[0]
#         predictions = np.argmax(pred, axis = 1).reshape(1, -1)
#         label = np.argmax(y, axis = 1).reshape(1, -1)
#         acc += np.sum((predictions == label) == 1)
#         np.savetxt('val_prediction.txt', predictions)
#     train_loss /= (n//batch_size)
#     acc = float(acc)
#     acc /= tol
#     return train_loss, acc
# def test(model, Xtest, Ytest):
#     model.val()
#     n = len(Xtest)
#     acc = 0
#     img = np.reshape(Xtest, (len(Xtest), 3, 32, 32))
#     pred = model(img)
#     predictions = np.argmax(pred, axis = 1).reshape(1, -1)
#     label = np.argmax(Ytest, axis = 1).reshape(1, -1)
#     acc = np.sum((predictions == label) == 1) 
#     acc = float(acc) / n
#     return acc

# if __name__ == '__main__':
#     batch_size = 64
#     conv_w = 5
#     conv_h = 5
#     conv_stride = 1
#     filter_num = 36
#     channel = 3
#     pad = 2
#     pool_w = 8
#     pool_h = 8
#     pool_stride = 8
#     lr = 0.01
#     momentum = 0.9
#     weight_decay = 0.001
#     seeds = [5]
#     train_time = len(seeds)
#     filename = 'sampledCIFAR10'
#     Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = loaddata(filename)
#     epoch = 100
#     H = 32
#     W = 32
#     num_hidden = 100
#     loss1 = np.zeros((epoch, 1))
#     acc1 = np.zeros((epoch, 1))
#     loss2 = np.zeros((epoch, 1))
#     acc2 = np.zeros((epoch, 1))
#     for j in range(len(seeds)):
#         print("training round "+str(j+1))
#         np.random.seed(seeds[j])
#         model = three_convoluted_neural_network(conv_w, conv_h, conv_stride, filter_num, channel, pad, pool_w, lr, momentum, weight_decay, H, W, num_hidden)
#         for i in range(epoch):
#             loss1[i], acc1[i] = train(model, Xtrain, Ytrain, batch_size)
#             loss2[i], acc2[i] = val(model, Xval, Yval, batch_size)
#             print i, loss1[i], acc1[i]
#         w_out = model.w_out
#         w_out = np.reshape(w_out, (36, -1))
#         np.savetxt('3bweight.txt', w_out)

