import numpy as np
import sys
import time
import matplotlib.pyplot as plt

def mse(ye, ya):
	sse = 0.0
	for i in range(0, len(ye)):
		e = ye[i]-ya[i]
		sse = sse + (e*e)
	mse = sse/len(ye)
	return mse

def maketargetmat(x):
	tar = np.zeros(10)
	tar[x] = 1
	return tar

def updwho(out, tar, sig1):
	return np.matmul(sig1.transpose(), out-tar)

def updwih(out, tar, who, sig1, inp):
	inp = inp.transpose()
	diff = (out-tar).transpose()
	sig1 = sig1*(1-sig1)
	err = who.dot(diff)
	exp = np.multiply(sig1, err.transpose())
	return inp.dot(exp)

def read(fname):
	f = open(fname, "r")
	print "Reading", fname 
	contents = f.read()
	f.close()
	splitlist = contents.split('[')
	finvec = []
	for i in splitlist:
		temp = []
		temp2 = []
		i = i.replace(']','').replace('\n','')
		temp = i.split(' ')
		for j in temp:
			if j != "":
				temp2.append((int(j))/255.0)
		nptemp = np.array(temp2)
		if nptemp.size > 0:
			finvec.append(nptemp)
	nptraindata = np.array(finvec)
	return nptraindata

def readlabels(fname):
	f = open(fname, "r")
	print "Reading", fname 
	contents = f.read()
	splitlist = contents.split('\n')
	trainlabellist = []
	for i in splitlist:
		if i != "":
			trainlabellist.append(int(i))
	nptrainlabel = np.array(trainlabellist)
	return nptrainlabel

def train(f, l, r):
	# t_rate = 0.0012
	begin = time.time()
	t_rate = r
	nptraindata = read(f)
	nptrainlabel = readlabels(l)

	print "Initializing random weights..."

	wih = 2*(np.random.rand(784, 30))-1
	who = 2*(np.random.rand(30, 10))-1
	plot_time = []
	plot_error = []
	start = time.time()

	for i in range(0,2):
		print "Running epoch", (i+1)
		errvec = []
		tvec = []
		for j in range(0,len(nptraindata)):
			wix = nptraindata[j].dot(wih)
			sig1 = 1/(1+np.exp(-wix))
			whx = sig1.dot(who)
			sig2 = 1/(1+np.exp(-whx))
			nptar = maketargetmat(nptrainlabel[j])
			error = mse(sig2, nptar)
			errvec.append(error)
			tvec.append(time.time()-start)
			sig1 = np.array([sig1])
			sig2 = np.array([sig2])
			nptar = np.array([nptar])
			wih = wih - (updwih(sig2, nptar, who, sig1, np.array([nptraindata[j]]) ))*t_rate 
			who = who - (updwho(sig2, nptar, sig1))*t_rate 
		print "Epoch", (i+1), "done"
		nin = 0
		for k in range(0,len(nptraindata)/1200):
			plot_time.append(tvec[nin])
			plot_error.append(( 1-errvec[nin])*100)
			nin = nin + 1200
	plt.plot(plot_time, plot_error)
	plt.ylabel("Accuracy Percentage")
	plt.xlabel("Time")
	plt.title("Accuracy vs Time for Learning Rate: " + str(t_rate))
	plt.show()
	w = open("netWeights.txt","w")
	w.write("WIH\n")
	np.savetxt(w,wih)
	w.write("\nWHO\n")
	np.savetxt(w,who)
	w.close()
	print "Training complete. \nTime taken:", round((time.time()-begin), 2), "seconds"

wih_test = []
who_test = []

def readweights(fname):
	global wih_test
	global who_test
	f = open(fname, "r")
	print "Reading", fname 
	contents = f.readline()
	if contents == "WIH\n":
		contents = f.readline()
	while contents != "WHO\n":
		contents = contents.split()
		contents = map(float, contents)
		if(contents != []):
			wih_test.append(contents)
		contents = f.readline()
	wih_test = np.array(wih_test)
	while contents != []:
		contents = f.readline()
		contents = contents.split()
		contents = map(float, contents)
		if(contents != []):
			who_test.append(contents)		
	who_test = np.array(who_test)

def test(f, l, w):
	global wih_test
	global who_test
	begin = time.time()
	nptestdata = read(f)
	nptestlabel = readlabels(l)
	h = 0
	readweights(w)
	for i in range(0, len(nptestdata)):
		m = np.dot(nptestdata[i], wih_test)
		m = 1/(1+np.exp(-m))
		o = np.dot(m, who_test)
		o = 1/(1+np.exp(-o))
		l = (np.where(o == max(o)))[0][0]
		if(l == nptestlabel[i]):
			h = h+1
	print "Testing complete."
	print "Hits:", h, "\nAccuracy:", (h*100.0)/(len(nptestlabel)), "\nTime taken:", round((time.time()-begin), 2), "seconds" 

def main():
	if len(sys.argv) != 5:
		print "Command line argument error. \nFor training, enter in the format: \npython MyNetwork.py train train.txt train-labels.txt learningRate \nFor testing, enter in the format: \npython MyNetwork.py test test.txt test-labels.txt netWeights.txt"
	if sys.argv[1] == "train":
		train(sys.argv[2], sys.argv[3], float(sys.argv[4]))
	if sys.argv[1] == "test":
		test(sys.argv[2], sys.argv[3], sys.argv[4])

main()