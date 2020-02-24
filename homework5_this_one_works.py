import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3312345)
class RNN:
    def __init__ (self, numHidden, numInput, numOutput):
        self.numHidden = numHidden
        self.numInput = numInput
        self.U = np.random.randn(numHidden, numHidden) * 1e-1
        self.V = np.random.randn(numHidden, numInput) * 1e-1
        self.w = np.random.randn(numInput,numHidden) * 1e-1


    def backward (self, y, yhats, xs, hs):
        T = len(y)
    
        gradient_V, gradient_U, gradient_w = np.zeros_like(self.V), np.zeros_like(self.U), np.zeros_like(self.w) # make all zero matrices.
        gradient_h_next = np.zeros_like(hs[0])

        for t in reversed(range(T)):
            gradient_MSE = (yhats[t] - y[t]) 

            gradient_w += np.dot(gradient_MSE, hs[t].T)
            gradient_h = np.dot(self.w.T, gradient_MSE) + gradient_h_next

            gradient_h_raw = ((1 - hs[t] ** 2)) * gradient_h
            
            gradient_V += np.dot(gradient_h_raw, xs[t].T)
            gradient_U += np.dot(gradient_h_raw, hs[t-1].T)

            gradient_h_next = np.dot(gradient_U.T, gradient_h_raw)

        return gradient_V, gradient_U, gradient_w

    def forward (self, x, y, pre_h):

        hs, yhats = {}, {}
        hs[-1] = np.copy(pre_h)
        loss = 0

        for t in range(len(x)):
            h_t = np.tanh(self.V.dot(x[t]) + np.dot(self.U, hs[t-1]))
            hs[t] = h_t
            self.w.dot(hs[t])
            yhats[t] = self.w.dot(hs[t])
            loss += (np.square(yhats[t] - y[t])) * 0.5

        return loss, yhats, hs, x   

    def SGD(self,x, y, batch_size, epochs, epsilon):      
            if len(epsilon) is not 3:
                raise ValueError("epsilon argument must be a list of real numbers and len() == 3") 

            for epoch in range(1, epochs+1):
                pre_h = np.zeros((self.numHidden,1)) 

                loss, yhats, hs, xs = self.forward(x, y, pre_h)
                gradient_V, gradient_U, gradient_w = self.backward(y,  yhats, x, hs)

                self.V -= epsilon[0] * gradient_V
                self.U -= epsilon[1] * gradient_U
                self.w -= epsilon[2] * gradient_w


                # MSE_val = MSE(np.array(list((self.yhats.values()))), y)
                if epoch % 100 == 0:
                    print("This is epoch round [{}]".format(epoch))
                    print("Epoch [{}], MSE Loss: {}".format(epoch , loss))

                if loss < 0.05:
                    early_stop +=1
                    if early_stop > 10:
                        print("This is epoch round [{}]".format(epoch))
                        print("Epoch [{}], MSE Loss: {}".format(epoch , loss))
                        return True
                else:
                    early_stop = 0


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


def MSE (yhat, y):
    number_samples = yhat.shape[0] # assumes yhat is (1,1)
    coeff = 1 / (2 * number_samples) 
    sum = np.sum((yhat - y)**2)
    mse = coeff * sum
    return mse

if __name__ == "__main__":
    xs, ys = generateData()
    numHidden = 6
    numInput = 1
    numTimesteps = len(xs)
    rnn = RNN(numHidden, numInput, 1)
    batch_size = 1
    epsilon = 1
    epochs = 50000
    rnn.SGD(xs, ys, batch_size, epochs, [4,1,1e-4])

