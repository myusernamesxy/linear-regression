training = []
with open('data/training.txt', 'r') as tf:
    data = tf.readlines()
    for line in data:
        line = line.split(',')
        lines = []
        lines.append(1.0)
        for i in range(len(line)):
            lines.append(float(line[i]))
        training.append(lines)
theta = [0, 0, 0, 0, 0]
tag = set()
sum1 = 0.0
sum2 = 0.0
sum3 = 0.0
sum4 = 0.0
for i in range(len(training)):
    sum1 = sum1 + training[i][1]
    sum2 = sum2 + training[i][2]
    sum3 = sum3 + training[i][3]
    sum4 = sum4 + training[i][4]
avg1 = sum1/len(training)
avg2 = sum2/len(training)
avg3 = sum3/len(training)
avg4 = sum4/len(training)
var1 = 0.0
var2 = 0.0
var3 = 0.0
var4 = 0.0
for i in range(len(training)):
    var1 += pow(training[i][1]-avg1, 2)
    var2 += pow(training[i][2]-avg2, 2)
    var3 += pow(training[i][3]-avg3, 2)
    var4 += pow(training[i][4]-avg4, 2)
var1 = var1/(len(training)-1)
var2 = var2/(len(training)-1)
var3 = var3/(len(training)-1)
var4 = var4/(len(training)-1)
dev1 = pow(var1, .5)
dev2 = pow(var2, .5)
dev3 = pow(var3, .5)
dev4 = pow(var4, .5)
for i in range(len(training)):
    training[i][1] = (training[i][1] - avg1) / dev1
    training[i][2] = (training[i][2] - avg2) / dev2
    training[i][3] = (training[i][3] - avg3) / dev3
    training[i][4] = (training[i][4] - avg4) / dev4


def gradientloss(theta):
    trainingloss = 0.0
    trainingdiff = dotfunc(training, theta)
    for i in range(len(trainingdiff)):
        trainingloss = trainingloss + pow(trainingdiff[i]-training[i][5], 2)
    trainingloss = trainingloss / (2 * len(training))
    return trainingloss


def dotfunc(training, theta):
    listmulti = []
    for i in range(len(training)):
        tag.add(training[i][5])
        val = 0.0
        for j in range(len(training[i]) - 1):
            val = val + training[i][j]*theta[j]
        listmulti.append(val)
    return listmulti


def dottrans(training, subval):
    alpha = 0.02
    listfinal = []
    for i in range(len(training[0])-1):
        val = 0.0
        for j in range(len(training)):
            val = val + training[j][i]*subval[j]
        listfinal.append(alpha*val/len(training))
    return listfinal


def gradientdescent(training, theta):
    step = 0
    while step <= 500:
        multi = dotfunc(training, theta)
        subval = []
        for i in range(len(training)):
            subval.append(multi[i] - training[i][5])
        trans = dottrans(training, subval)
        for j in range(len(theta)):
            theta[j] = theta[j] - trans[j]
        step = step+1
    return theta


theta = gradientdescent(training, theta)
print("The prediction values of theta are: ", theta)
print()
print("Estimate function is: y =", theta[0], "+ (", theta[1], ") * x1 + (", theta[2],
      ") * x2 +", theta[3], "* x3 +", theta[4], "*x4")
testing = []
with open('data/testing.txt', 'r') as testf:
    datatest = testf.readlines()
    for line in datatest:
        line = line.split(',')
        lines = []
        lines.append(1.0)
        for i in range(len(line)):
            lines.append(float(line[i]))
        testing.append(lines)


sumt1 = 0.0
sumt2 = 0.0
sumt3 = 0.0
sumt4 = 0.0
for i in range(len(testing)):
    sumt1 = sumt1 + testing[i][1]
    sumt2 = sumt2 + testing[i][2]
    sumt3 = sumt3 + testing[i][3]
    sumt4 = sumt4 + testing[i][4]
avgt1 = sumt1/len(testing)
avgt2 = sumt2/len(testing)
avgt3 = sumt3/len(testing)
avgt4 = sumt4/len(testing)
vart1 = 0.0
vart2 = 0.0
vart3 = 0.0
vart4 = 0.0
for i in range(len(testing)):
    vart1 += pow(testing[i][1]-avgt1, 2)
    vart2 += pow(testing[i][2]-avgt2, 2)
    vart3 += pow(testing[i][3]-avgt3, 2)
    vart4 += pow(testing[i][4]-avgt4, 2)
vart1 = vart1/(len(testing)-1)
vart2 = vart2/(len(testing)-1)
vart3 = vart3/(len(testing)-1)
vart4 = vart4/(len(testing)-1)
devt1 = pow(vart1, .5)
devt2 = pow(vart2, .5)
devt3 = pow(vart3, .5)
devt4 = pow(vart4, .5)
for i in range(len(testing)):
    testing[i][1] = (testing[i][1] - avgt1) / devt1
    testing[i][2] = (testing[i][2] - avgt2) / devt2
    testing[i][3] = (testing[i][3] - avgt3) / devt3
    testing[i][4] = (testing[i][4] - avgt4) / devt4


def calculateloss():
    diff = dotfunc(testing, theta)
    sumdiff = 0.0
    for i in range(len(diff)):
        sumdiff = sumdiff + pow(diff[i]-testing[i][5], 2)
    loss = sumdiff / (2 * len(testing))
    return loss


sumloss = calculateloss()
print()
print("The loss of the estimation function to the training set is: ", gradientloss(theta))
print()
print("The loss of the estimation function to the testing set is ", sumloss)


def calcorrect(theta, testing):
    correctnum = 0
    wrongnum = 0
    taglist = []
    for num in range(len(tag)):
        taglist.append(tag.pop())
    for i in range(len(testing)):
        diff = theta[0] + theta[1] * testing[i][1] + theta[2] * testing[i][2] + \
               theta[3] * testing[i][3] + theta[4] * testing[i][4]
        min = abs(diff - taglist[0])
        flag = taglist[0]
        for j in range(1, len(taglist)):
            if abs(diff - taglist[j]) < min:
                min = abs(diff - taglist[j])
                flag = taglist[j]
        if flag == testing[i][5]:
            correctnum += 1
        else:
            wrongnum += 1
    correctrate = correctnum / (correctnum+wrongnum)
    return correctrate


print()
print("Accuracy rate is:", calcorrect(theta, testing))