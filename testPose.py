#! /usr/bin/python
import caffe
import numpy as np
import matplotlib.pyplot as plt
from read_binaryproto import *
import lmdb

# set raw scale!!!!!
# this is because caffe.io.load_image get the image scale within [0,1] but
# the network use the one within [0,255] !!!
# this cause the predict result to be false!!!


def testNet_imagePath(protoPath,modelSnapshot,imgPath):
	net = caffe.Net(protoPath,modelSnapshot,caffe.TEST)
	img = caffe.io.load_image(imgPath)
	imgSize = img.shape
	inputSize = net.blobs['data'].data.shape[1:]
	f, (ax1,ax2) = plt.subplots(1,2)
	ax1.imshow(img)
	ax2.imshow(img)
	#img = img.transpose(2,0,1)
	img = img.swapaxes(0, 2).swapaxes(1, 2)
	img = caffe.io.resize(img,inputSize)
	#print imgSize
	#print img.shape
	mean = read_binaryproto('/home/xacti-dnn1/HDD/DNN/deeppose/data/image_mean.binaryproto')
	net.blobs['data'].data[0] = img*255 - mean
	output = net.forward(['conv5','predict'])
	#print output['conv5'].shape
	output = output['predict'][0]
	
	truth = np.load('test.npy')
	#print "truth:",truth
	xIdx = range(0,truth.shape[0],2)
	yIdx = range(1,truth.shape[0],2)

	xArr = truth[xIdx] * imgSize[0]
	yArr = truth[yIdx] * imgSize[1]
	#print xArr,yArr
	
	xOut = output[xIdx] * imgSize[0]
	yOut = output[yIdx] * imgSize[1]

	ax1.scatter(xArr, yArr)
	ax1.plot(xArr, yArr)
	
	ax2.scatter(xOut, yOut)
	ax2.plot(xOut, yOut)
	print calLoss(output,truth)
	plt.show()

def testNet_imageArr(protoPath,modelSnapshot,img,truth):
	net = caffe.Net(protoPath,modelSnapshot,caffe.TEST)
	imgSize = img.shape
	inputSize = net.blobs['data'].data.shape[1:]
	f, (ax1,ax2) = plt.subplots(1,2)
	ax1.imshow(img)
	ax2.imshow(img)
	img = img.transpose(2,0,1)
	img = caffe.io.resize(img,inputSize)
	mean = read_binaryproto('/home/xacti-dnn1/HDD/DNN/deeppose/data/image_mean.binaryproto')
	net.blobs['data'].data[0] = img*255 - mean
	output = net.forward()['predict'][0]
	print output
	
	if not truth == False:
		print "truth:",truth
		xIdx = range(0,truth.shape[0],2)
		yIdx = range(1,truth.shape[0],2)

		xArr = truth[xIdx] * imgSize[0]
		yArr = truth[yIdx] * imgSize[1]
		print xArr,yArr
		ax1.scatter(xArr, yArr)
		ax1.plot(xArr, yArr)
			
	xOut = output[xIdx] * imgSize[0]
	yOut = output[yIdx] * imgSize[1]
	ax2.scatter(xOut, yOut)
	ax2.plot(xOut, yOut)
	print calLoss(output,truth)
	plt.show()

def calLoss(x1,x2):
	diff = x1-x2
	square = [ii*ii for ii in diff ]
	return sum(square)/2

def testFrom_lmdb(protoPath,modelSnapshot,image_lmdb,label_lmdb):
    # load lmdb data and label
	lmdb_img_env = lmdb.open(image_lmdb)
	lmdb_label_env = lmdb.open(label_lmdb)
	lmdb_img_txn = lmdb_img_env.begin()
	lmdb_label_txn = lmdb_label_env.begin()
	lmdb_img_cursor = lmdb_img_txn.cursor()
	lmdb_label_cursor = lmdb_label_txn.cursor()
	datum_img = caffe.proto.caffe_pb2.Datum()
	datum_label = caffe.proto.caffe_pb2.Datum()
	# randomly set the cursor
	import random
	iterNum = int(random.random()*1000)
	for ii in xrange(iterNum):
		lmdb_img_cursor.next()
	key = lmdb_img_cursor.key()

	datum_label.ParseFromString(lmdb_label_cursor.get(key))
	datum_img.ParseFromString(lmdb_img_cursor.get(key))
	truth = caffe.io.datum_to_array(datum_label)
	img = caffe.io.datum_to_array(datum_img)

	net = caffe.Net(protoPath,modelSnapshot,caffe.TEST)
	imgSize = img.shape
	print imgSize
	inputSize = net.blobs['data'].data.shape[1:]
	f, (ax1,ax2) = plt.subplots(1,2)
	# for plot
	img_plot = img.transpose(1,2,0)
	ax1.imshow(img_plot)
	ax2.imshow(img_plot)

	img = caffe.io.resize(img,inputSize)
	mean = read_binaryproto('/home/xacti-dnn1/HDD/DNN/deeppose/data/image_mean.binaryproto')
	net.blobs['data'].data[0] = img*255 - mean
	output = net.forward()['predict'][0]
	#print output
	
	#print "truth:",truth
	xIdx = range(0,truth.shape[0],2)
	yIdx = range(1,truth.shape[0],2)

	xArr = truth[xIdx,0,0] * imgSize[1]
	yArr = truth[yIdx,0,0] * imgSize[2]
	#print xArr,yArr
	
	xOut = output[xIdx] * imgSize[1]
	yOut = output[yIdx] * imgSize[2]

	ax1.scatter(xArr, yArr)
	ax1.plot(xArr, yArr)
	
	ax2.scatter(xOut, yOut)
	ax2.plot(xOut, yOut)
	print calLoss(output,truth)
	plt.show()
	
	

if __name__ == "__main__":
	#testNet_imagePath('/home/xacti-dnn1/HDD/DNN/deeppose/models/AlexNet/predict.prototxt','/home/xacti-dnn1/HDD/DNN/deeppose/snapshots/AlexNet_iter_10000.caffemodel','baseball.jpg')
	#testNet_imagePath('/home/xacti-dnn1/HDD/DNN/deeppose/models/AlexNet/predict.prototxt','/home/xacti-dnn1/HDD/DNN/deeppose/snapshots/AlexNet_iter_10000.caffemodel','baseball1.jpg')
	#testNet_imagePath('/home/xacti-dnn1/HDD/DNN/deeppose/models/AlexNet/predict.prototxt','/home/xacti-dnn1/HDD/DNN/deeppose/snapshots/AlexNet_iter_10000.caffemodel','baseball2.jpg')
	#testNet_imagePath('/home/xacti-dnn1/HDD/DNN/deeppose/models/AlexNet/predict.prototxt','/home/xacti-dnn1/HDD/DNN/deeppose/snapshots/AlexNet_iter_10000.caffemodel','baseball3.jpg')
	lmdb_img_path = '/home/xacti-dnn1/HDD/DNN/deeppose/data/test_image_train.lmdb'
	lmdb_label_path = '/home/xacti-dnn1/HDD/DNN/deeppose/data/test_joint_train.lmdb'
	testFrom_lmdb('/home/xacti-dnn1/HDD/DNN/deeppose/models/AlexNet/predict.prototxt','/home/xacti-dnn1/HDD/DNN/deeppose/snapshots/AlexNet_iter_10000.caffemodel',lmdb_img_path,lmdb_label_path)
	
