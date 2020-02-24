import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize  # For check_grad, approx_fprime
import math
from scipy.optimize import check_grad

np.random.seed(1234)

# return list of tuples (start,end) for slicing each batch X
def get_indexes(n, batchSize):
    indexes = []  # list of (start,end) for slicing each batch X
    index = 0
    for round in range(math.ceil(n / batchSize)):
        index += batchSize
        if index > n:
            index -= batchSize
            indexes.append((index, n))
            break
        indexes.append((index - batchSize, index))
    return indexes

class RNN:
    def __init__ (self, numHidden, numInput, numOutput):
        self.numHidden = numHidden
        self.numInput = numInput
        self.U = np.random.randn(numHidden, numHidden) * 1e-1
        self.V = np.random.randn(numHidden, numInput) * 1e-1
        self.w = np.random.randn(numHidden) * 1e-1
        self.hs = [np.zeros((numHidden, numInput))]
        # TODO: IMPLEMENT ME

    def backward (self,x, y):
        # TODO: IMPLEMENT ME
        yhat = self.forward(x)
        T = len(y) # = 50
        # print(self.yhats.keys())
        gradient_MSE = yhat[-1] - y[-1] # y = pos(50) - pos(49)

        #init gradient values
        # gradient_w = np.zeros(self.w.shape).reshape(-1,1)
        # gradient_U = np.zeros(self.U.shape)
        # gradient_V = np.zeros(self.V.shape)

        gradient_w = 0
        gradient_U = 0
        gradient_V = 0


        gradient_h = np.dot(gradient_w, gradient_MSE).reshape(-1,1)
        
        for t in np.arange(T)[::-1]:
            # gradient_MSE = (yhat[t] - y[t])
            # print("temp stuff")
            # print(gradient_h.shape)
            # temp = (gradient_h * (1 - self.hs[t+1] ** 2))
            # print(temp.shape)

            # print(f"gradient_h before is {gradient_h.shape}")
            temp = np.dot(gradient_h,np.diag(1 - self.hs[t+1] ** 2))
            # print(temp.shape)

            gradient_w += np.dot(gradient_MSE, self.hs[t+1].T).reshape(-1,1) # could be self.hs[t-1]

            # print("gradien U")
            # print(temp.shape)
            gradient_U += temp.reshape(-1,1).dot(np.transpose(self.hs[t]))
            # gradient_U += 
            gradient_V += temp.reshape(-1,1).dot(np.transpose(self.last_x[t]))

            gradient_MSE = np.array(yhat[t-1] - y[t-1])
            gradient_h = self.U.dot(temp).reshape(-1,1) #+  np.dot(np.transpose(gradient_w), gradient_MSE)
            # print(f"gradient_h is {gradient_h.shape}")
        
        # try wth out clip and see if it works!!!! study a little more the aspects of the algo.
        # for gradint in [gradient_V, gradient_U, gradient_w]:
        #     np.clip(gradint, -1, 1, out=gradint)

        return gradient_V, gradient_U, gradient_w

    def forward (self, x):
        # TODO: IMPLEMENT ME

        self.last_x = x
        self.hs = [np.zeros((numHidden, numInput))]
        yhats = []

        for i in range(len(x)):
            h_t = np.tanh(self.V.dot(x[i]) + self.U.dot(self.hs[i]))
            yhat_t = self.w.dot(h_t)
            self.hs.append(h_t)
            yhats.append(yhat_t)

        # return last saved yhat & h time step 
        return yhats 

    def SGD(self,x, y, batch_size, epochs, epsilon):
        # randomize training set

        rate1 = 1e-1
        rate2 = 1e-5
        rate3 = 1e-7


        # start iteration loop
        for epoch in range(1, epochs+1):
            # print(f"Epoch [{epoch}]")
            gradient_V, gradient_U, gradient_w = self.backward(x, y)
            # self.V -= rate3 * gradient_V
            # self.U -= rate2 * gradient_U
            # self.w -= rate1 * gradient_w.reshape(-1)
            self.V -= epsilon * gradient_V
            self.U -= epsilon * gradient_U
            self.w -= epsilon * gradient_w.reshape(-1)

            yhat = self.forward(x)
            MSE_val = MSE(yhat, y)
            print("This is epoch round [{}]".format(epoch))
            print("Epoch [{}], MSE Loss: {}".format(epoch , MSE_val))

def MSE (yhat, y):
    return 0.5 * np.sum((yhat - y)**2)

# From https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
def generateData ():
    total_series_length = 50
    echo_step = 2  # 2-back task
    batch_size = 1
    x = np.random.choice(2, total_series_length, p=[0.5, 0.5])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    y = list(y)
    return (x, y)



if __name__ == "__main__":
    xs, ys = generateData()
    numHidden = 6
    numInput = 1
    numTimesteps = len(xs)
    rnn = RNN(numHidden, numInput, 1)

    batch_size = 1
    epsilon = 0.1
    epochs = 1000000

    # a= check_grad(forward,backward,y)
    # print(a)
    rnn.SGD(xs, np.array(ys), batch_size, epochs,epsilon)




    # rnn2 = rnn.forward(np.array(xs).reshape(1,-1))
    # print(rnn.backward(ys))

    # print(rnn.yhats[50])
    # TODO: IMPLEMENT ME

