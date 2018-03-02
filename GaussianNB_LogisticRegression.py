import numpy as np
import random
import math
from matplotlib import pyplot as plt

# Calculates mean for the given column
def calcMean(column_vector):
	total = sum(column_vector)
	count = len(column_vector)
	return total/count, count

# calculates Variance for the given column
def calcVariance(column_vector, mean, count):
	tot_sum = 0
	for element in column_vector:
		tot_sum += pow(element - mean,2)
	variance = tot_sum/float(count-1)
	return variance

# Gaussian Naive Bayes function to calculate mean and variance of train data to train the classifier
def GaussianNaiveBayes_train_test(training_data, testing_data, actual_y):
	myu_z,var_z, myu_o, var_o = [],[],[],[]
	train_zero, train_one = [], []
	count_tr_o, count_tr_z = 0, 0

	# splitting the training data according to labels
	for row in range(len(training_data)):   
		if training_data[row][-1]==0:
			train_zero.append(training_data[row])
		else:
			train_one.append(training_data[row])

	train_zero = np.array(train_zero)
	train_one = np.array(train_one).reshape(len(train_one),5)
	prior_z =  len(train_zero)/len(training_data) #finding priors for y=0
	prior_o =  len(train_one)/len(training_data) #finding priors for y=1
	for col in range(training_data.shape[1]-1):
		mean, count  = calcMean(train_zero[:,col])
		variance = calcVariance(train_zero[:,col], mean, count)
		myu_z.append(mean)
		var_z.append(variance)

	for col in range(training_data.shape[1]-1):
		mean, count  = calcMean(train_one[:,col])
		variance = calcVariance(train_one[:,col], mean, count)
		myu_o.append(mean)
		var_o.append(variance)

	def conditionalProb(dt, mean, vr):
		dr_term = 2*vr*math.pi
		mul_term = 1/math.sqrt(dr_term)
		epow = math.exp(-(np.power(dt-mean,2)/(2*vr)))
		prob = mul_term*epow
		return prob

	def calcLikelihood(x, m, v): # calculating the likelihood
		liklelihood = conditionalProb(x[0], m[0], v[0])*conditionalProb(x[1], m[1], v[1])*conditionalProb(x[2], m[2], v[2])*conditionalProb(x[3], m[3], v[3])
		return liklelihood

	def calcygivenx_posterior(lk_o, lk_z, p1, p0):
		p_ygivenx_o = (lk_o*p1)/((lk_o*p1)+(lk_z*p0))
		p_ygivenx_z = (lk_z*p0)/((lk_o*p1)+(lk_z*p0))
		return p_ygivenx_o, p_ygivenx_z


	x_yone_lk = []
	x_yzero_lk= []
	y1given_x=[]
	y0given_x=[]
	y_predict = []
	for row in range(len(testing_data)):
		x_yone_lk.append(calcLikelihood(testing_data[row], myu_o, var_o))
		x_yzero_lk.append(calcLikelihood(testing_data[row], myu_z, var_z))
		a,b= calcygivenx_posterior(x_yone_lk[row],x_yzero_lk[row],prior_o,prior_z)
		y1given_x.append(x_yone_lk[row]*a)
		y0given_x.append(x_yzero_lk[row]*b)
	x_yone_lk = np.array(x_yone_lk)
	x_yzero_lk= np.array(x_yzero_lk)
	y1given_x= np.array(y1given_x)
	y0given_x= np.array(y0given_x)
	for i in range(len(y0given_x)):
		y_predict.append(1 if y1given_x[i]>y0given_x[i] else 0)
	success = 0
	for i in range(len(y_predict)):
		if y_predict[i]-actual_y[i] == 0:
			success=success+1
	accuracy = (success/len(y_predict))
	return accuracy*100

def update_weights(sig_y,training_data,learning_rate,training_y,weights):
	training_y = training_y.reshape(len(training_y),1)
	grd = (np.dot(np.transpose(training_y-sig_y),training_data[:,:-1]))/len(training_data)
	update = np.transpose(grd)*learning_rate
	weights += update
	return weights

def findAccuracy(predicted_y,actual_y):
	s=0
	actual_y = actual_y.reshape(len(predicted_y),1)
	predicted_y= predicted_y.reshape(len(predicted_y),1)
	diff = actual_y-predicted_y
	for i in range(len(diff)):
		if diff[i]==0:
			s +=1
	return (s/len(predicted_y))*100

def logisticRegression(training_data, testing_data, learning_rate, training_y,actual_y):
	x0 = np.ones((len(training_data),1))
	training_data = np.hstack((x0,training_data))
	logreg_weights = np.random.rand(5,1)*0.03

	for i in range(6000):
		theta = np.dot(training_data[:,:-1], logreg_weights)
		sig_y = 1/(1+np.exp(-(theta)))
		logreg_weights = update_weights(sig_y,training_data,learning_rate,training_y,logreg_weights)
		
	x0_test = np.ones((len(testing_data),1))
	testing_data = np.hstack((x0_test,testing_data))
	theta_test = np.dot(testing_data[:,:-1], logreg_weights)
	sig_y_test = 1/(1+np.exp(-(theta_test)))
	y_p = []
	for y in range(len(sig_y_test)):
		y_p.append(0 if sig_y_test[y]<0.5 else 1)
	accuracy = findAccuracy(np.array(y_p),actual_y)
	return accuracy

def generate_points(train1, train2):
	train_12 = np.vstack((train1,train2))
	templist =[]
	for i in range(len(train_12)):   
		if train_12[i][-1]==1:
			templist.append(train_12[i])
	temparray = np.array(templist)
	m, v,k =[],[],[]
	for col in range(temparray.shape[1]-1):
		mean, count  = calcMean(temparray[:,col])
		variance = calcVariance(temparray[:,col], mean, count)
		m.append(mean)
		v.append(variance)
	print("mean for the training set is:",m)
	print("variance for the training set is:",v)
	for i in range(len(m)):
		k.append(np.random.normal(m[i],math.sqrt(v[i]),400))
	k=np.array(k)
	k = np.transpose(k)
	km,kv = [], []
	for col in range(k.shape[1]):
		mean, count  = calcMean(k[:,col])
		variance = calcVariance(k[:,col], mean, count)
		km.append(mean)
		kv.append(variance)
	print("mean for the generated set is:",km)
	print("variance for the generated set is:",kv)


datasetmatrix = np.loadtxt('dataset.txt',dtype= float,delimiter=',')

np.random.shuffle(datasetmatrix)
shape1, shape2 = datasetmatrix.shape
lensplit = int(0.67*shape1)
lensplit3 = int(0.33*shape1)
train_datamat, test_datamat = datasetmatrix[:lensplit],datasetmatrix[lensplit:]
trainsetA, trainsetB, trainsetC = datasetmatrix[:lensplit3], datasetmatrix[lensplit3:lensplit], datasetmatrix[lensplit:]

actual_y = test_datamat[:,-1]
random_frac = [.01, .02, .05, .1, .625, 1] #fraction of data to be considered
iterations = 5
log_avg_acc,gnb_avg_acc =[],[]

# running the given fraction of data for five iterations and averaging the result over 5 runs
for fc in random_frac: 
	a = 0
	for itr in range(iterations):
		a += logisticRegression(train_datamat[:int(fc*train_datamat.shape[0])], test_datamat, 0.02, train_datamat[:int(fc*train_datamat.shape[0]),-1],actual_y )
	log_avg_acc.append(a/iterations)
	print(a/iterations)
print("Accuracy for Logistic Regression over 5 runs for each fraction:",log_avg_acc)

# running the given fraction of data for five iterations and averaging the result over 5 runs
for fc in random_frac: 
	a = 0
	for itr in range(iterations):
		split_lmt = int(fc*train_datamat.shape[0])
		a += GaussianNaiveBayes_train_test(train_datamat[:split_lmt], test_datamat,actual_y)
	gnb_avg_acc.append(a/iterations)
	print(a/iterations)
print("Accuracy for GNB over 5 runs for each fraction:",gnb_avg_acc)

generate_points(trainsetA,trainsetB)
generate_points(trainsetA,trainsetC)
generate_points(trainsetB,trainsetC)


fgr,a = plt.subplots() 
plt.title("Accuracies of GNB and LR")
plt.xlabel("size of trainset")
plt.ylabel("accuracy")
a.plt(random_frac,gnb_avg_acc,'+-',label="GNB")
a.plt(random_frac,log_avg_acc,'*-',label="LR")
a.legend()
plt.show()

