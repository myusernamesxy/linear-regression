import numpy as np
A = []
Y = []
tag = set()
with open('data/training.txt', 'r') as tf:
    data = tf.readlines()
    for line in data:
        line = line.split(',')
        lines = []
        lines.append(1.0)
        for i in range(len(line)-1):
            lines.append(float(line[i]))
        A.append(lines)
        Y.append(float(line[len(line)-1]))
        tag.add(float(line[len(line)-1]))
taglist = []
for num in range(len(tag)):
    taglist.append(tag.pop())
A = np.array(A)
Y = np.array(Y)
D = np.dot(A.T, A)
D = np.mat(D)
parameterMatrix = np.dot(np.dot(D.I, A.T), Y)
print("The prediction values of theta are: ", parameterMatrix)
print()
print("Estimate function is: y =", parameterMatrix[0, 0], "+ (", parameterMatrix[0, 1], ") * x1 + (",
      parameterMatrix[0, 2], ") * x2 +", parameterMatrix[0, 3], "* x3 +", parameterMatrix[0, 4], "*x4")
trainError = 0
for i in range(len(A)):
    trainError = trainError + pow(Y[i] - parameterMatrix[0, 0] - parameterMatrix[0, 1] * A[i][1] -
                                  parameterMatrix[0, 2] * A[i][2] - parameterMatrix[0, 3] * A[i][3] -
                                  parameterMatrix[0, 4] * A[i][4], 2)
trainError = trainError / len(A)
print()
print("The loss of the estimation function to the training set is ", trainError)
test = []
with open('data/testing.txt', 'r') as testf:
    datatest = testf.readlines()
    for line in datatest:
        line = line.split(',')
        lines = []
        for i in range(len(line)):
            lines.append(float(line[i]))
        test.append(lines)
test = np.array(test)
sumError = 0
for i in range(len(test)):
    sumError = sumError + pow(test[i][4] - parameterMatrix[0, 0] - parameterMatrix[0, 1] * test[i][0] -
                              parameterMatrix[0, 2] * test[i][1] - parameterMatrix[0, 3] * test[i][2] -
                              parameterMatrix[0, 4] * test[i][3], 2)
sumError = sumError / len(test)
print()
print("The loss of the estimation function to the testing set is ", sumError)


def calcorrect(parameterMatrix, test):
    correctnum = 0
    wrongnum = 0
    for i in range(len(test)):
        diff = parameterMatrix[0, 0] + parameterMatrix[0, 1] * test[i][0] + parameterMatrix[0, 2] * test[i][1] + \
               parameterMatrix[0, 3] * test[i][2] + parameterMatrix[0, 4] * test[i][3]
        min = abs(diff - taglist[0])
        flag = taglist[0]
        for j in range(1, len(taglist)):
            if abs(diff - taglist[j]) < min:
                min = abs(diff - taglist[j])
                flag = taglist[j]
        if flag == test[i][4]:
            correctnum += 1
        else:
            wrongnum += 1
    correctrate = correctnum / (correctnum+wrongnum)
    return correctrate


print()
print("Accuracy rate is:", calcorrect(parameterMatrix, test))