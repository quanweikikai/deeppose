import caffe
import numpy as np
import matplotlib.pyplot as plt

def read_binaryproto(filePath):

	blob = caffe.proto.caffe_pb2.BlobProto()
	data = open( filePath , 'rb' ).read()
	blob.ParseFromString(data)
	arr = np.array( caffe.io.blobproto_to_array(blob) )
	out = arr[0]

	return out

#if __name__ == "__main__":
#	img = read_binaryproto('/home/xacti-dnn1/HDD/DNN/deeppose/data/image_mean.binaryproto')
#	print img.shape
#	img = img.transpose(1,2,0)
#	img = np.ones(img.shape)
#	img = caffe.io.load_image('test.jpg')
#	print img.shape
#	plt.imshow(img)
#	plt.show()
