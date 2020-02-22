import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize  # For check_grad, approx_fprime

'''
- For SGD, are all the batch sizes 1?
- what do we do with the hs (is it fine to restart it)

'''

np.random.seed(1234)

class RNN:
    def __init__ (self, numHidden, numInput, numOutput):
        self.numHidden = numHidden
        self.numInput = numInput
        self.U = np.random.randn(numHidden, numHidden) * 1e-1 # (6,6)
        self.w = np.random.randn(numHidden) * 1e-1 # (6,1)
        # TODO: IMPLEMENT ME
        self.h = [np.zeros(6)] # length of 51 since x is 50 numbers + initial h = 0
        self.V = np.random.randn(numHidden, numInput) * 1e-1 # (6,1)

    def backward (self, x, y):
        # TODO: IMPLEMENT ME
        T = len(y)

        yhat = self.forward(x)

        # initialize gradients
        delta_u = np.zeros(self.U.shape)
        delta_v = np.zeros(self.V.shape)

        # calculate delta w
        delta_w = np.dot(yhat[-1]-y[-1],self.h[-1].T)

        q = np.dot(yhat[-1] - y[-1], self.w.T).T
        g = 1 - self.h[-1]**2 # this just represents one h, not the array of hs
        # n = np.arange(T)[-1]
        # g = 1 - self.h[n+1]**2
        for t in np.arange(T)[::-1]:
            q = q * g.T

            delta_u += np.dot(q, self.h[t]) # t = t -1, because our index starts at 49, not 50
            
            print(f"the shape of q is {q.shape}")
            print(f"the shape of x is {x[t-1].shape}")
            print(f"current t is {t}")
            # current proble is that x is an int and doing dot prduct is getting destroyed
            delta_v += np.dot(q, x[t-1]) # todo: what index will this index for the last element?, i.e. if t= 0, will this try to index the last x?

            # update g from t to t-1 for next loop
            g = 1 - self.h[t-1]**2 # index the t-1 elementh (extra -1 because how python indexing works -1 = last)
            # the t-1 element, will be just t since our h has 51 elements instead of 50 (last index is 50)
            q = np.dot(q.T,self.U) * g.T

        print(f"my delta u shape is {delta_u.shape}")
        print(f"my delta v shape is {delta_v.shape}")
        return delta_w, delta_u, delta_v

    def forward (self, x):
        # TODO: IMPLEMENT ME

        yhat = [] # all my guesses, should be 50
        self.h = [np.zeros(6)] # restart memory

        # z = self.U @ h[step-1] + self.V @ xt
        # ht = tanh(z)
        # yhat.append(ht.T @ self.w)

        # iterate 50 times, start at 1th index 
        for t in range(len(x)):
            z = np.dot(self.U, self.h[t]) + np.dot(self.V, x[t]) # 
            ht = np.tanh(z)
            self.h.append(ht)
            yhat.append(np.dot(ht.T, self.w))

        return yhat


# MSE for the complete data set
def MSE (yhat, y):
    # yhat = np.dot(X_tilde.T, w)
    number_samples = yhat.shape[0] # assumes yhat is (1,1)
    coeff = 1 / (2 * number_samples) 
    sum = np.sum((yhat - y)**2)
    mse = coeff * sum
    return mse

def grad_tanh(z):
    return 1 - np.tanh(z)**2


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
    
    rnn.backward(xs, ys)

    # yhat = rnn.forward(xs)
    # print(yhat)
    # print(len(yhat))

    # TODO: IMPLEMENT ME

    # print(rnn.w.shape)
    # print(rnn.U.shape)
    # print(rnn.V.shape)