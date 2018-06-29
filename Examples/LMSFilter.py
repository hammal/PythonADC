import numpy as np
from AnalogToDigital import LeastMeanSquare

data = np.load("data.npy")[()]
FIRLenght = 256
midPoint = FIRLenght >> 1
## Arrange data
x = data["x"].transpose()
y = data["y"]
trainingX = np.zeros((x.shape[0],FIRLenght * x.shape[1]))
trainingY = np.zeros(x.shape[0])
trainingY[midPoint::] = y[:-midPoint]

for index in range(midPoint, x.shape[0]-midPoint):
    trainingX[index - midPoint, :] = np.reshape(x[index-midPoint:index + midPoint, :], FIRLenght * x.shape[1])

lms = LeastMeanSquare()
lms.train({"y":trainingY, "x":trainingX})

## and the training error
y_hat = np.apply_along_axis(lms.predict, axis=1, arr=trainingX)
error = y - y_hat
print("MSE on training data = %s" % (np.linalg.norm(error)/error.size))
