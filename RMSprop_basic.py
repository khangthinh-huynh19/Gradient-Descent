import numpy as np
import math


"In this RMSprop code, I will apply it to the Binary Cross Entropy Loss fucntion"

def LogLoss(y_true,y_pre):
    "The Binary Cross Entropy fucntion "
    epsilon = 1e-15
    y_predicted_new = [max(i,epsilon) for i in y_pre]
    y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))


def sigmoid(X,w,b):
    "The sigmoid function"
    y=1/ (1+np.exp(-np.dot(X,w)-b))
    return y


def RMSprop(X_data,y_true,beta=0.9,eta=0.01,epsilon=1e-8,epochs=200):
    #Initialize the W and b
    "In this code, I let both W and b starts with 1"
    W=np.ones((X_data.shape[1],1))
    b=np.ones((1,1))
    n=X_data.shape[0]
    #Initialize for SdW and Sdb
    SdW=np.ones(W.shape)
    Sdb=np.ones(b.shape)

    "Using full-batch GD for RMSprop"
    for epoch in range(1,epochs+1):
        y_predicted=sigmoid(X_data,W,b)
        loss=LogLoss(y_true,y_predicted)
        grad_W=(1/n)*np.dot(X_data.T,(y_predicted-y_true))
        grad_b=(1/n)*np.sum(y_predicted-y_true)

        "Updating the SdW and SdB"
        SdW=beta*SdW+(1-beta)*(np.square(grad_W))
        Sdb=beta*Sdb+(1-beta)*(np.square(grad_b))

        "Gradient Descent on W and b"
        W=W-eta*(grad_W/ (np.sqrt(SdW)+epsilon))
        b=b-eta*(grad_b/ (np.sqrt(Sdb)+epsilon))

        if epoch%10==0:
            print('Epoch: {}, weights: {}, bias: {}, loss: {}'.format(epoch,W,b,loss))

    return W,b 


"""Note: This is just the basic RMSprop code, you can customize it with different Loss/Cost functions,
 epochs, beta, Initial Value of W,b,Sdw,Sdb (with different method)
 You can increase the speed by uing mini-batch, SGD."""

"""This is just the simple trainning data, you can test its performance with larger data"""
X=np.array([[0.8,0.2,0.3],[0.7,0.5,0.4]])
y=np.array([[0],[1]])
W,b=RMSprop(X,y)
