import caffe
import numpy as np
import yaml
import cv2

DEBUG = False
kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])

class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
		top[0].reshape(bottom[0].data.shape[0],14,1,1)

        self.kernel_size = layer_params['kernel_size']
		assert self.kernel_size%2 == 1, "kernel size must be odd!"
        
        if DEBUG:
        #if True:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors
	
	def pdf(self,point,mean,cov):
	  	return cons*np.exp(-np.dot(np.dot((point-mean),np.linalg.inv(cov)),(point-mean).T)/2.)

	def add_pad(self,dataArr):
		kernelSize = int(self.kernel_size/2)*2
		outputArr = np.zeros((dataArr.shape[0]+int(self.kernel_size/2)*2,dataArr.shape[1]+int(self.kernel_size/2)*2))
		for ii in xrange(dataArr.shape[0]):
			for jj in xrange(dataArr.shape[1]):
				outputArr[kernelSize+ii,kernelSize+jj] = dataArr[ii,jj]
		return outputArr

	def arr2coordinate(self,dataArr):
		assert len(dataArr.shape) == 2, "data must be 2 dimension"
		height = dataArr.shape[0]
		width = dataArr.shape[1]
		p_min = np.min(dataArr)
		# add offset to make every pixel value bigger than 0
		dataArr = dataArr - p_min
		filteredArr = cv2.filter2d(dataArr,-1,kernel)
		idx = np.argmax(filteredArr)
		idx_x = idx / filteredArr.shape[0]
		idx_y = idx % filteredArr.shape[0]
		#add 0 pad to blob
		dataArr = self.add_pad(dataArr)
		x_sum, y_sum = 0
		#idx_x idx_y add 1 after padded
		idx_x += 1
		idx_y += 1
		#pixel sum
		p_sum = np.sum(dataArr[idx_x-1:idx_x+2,idx_y-1:idx_y+2])
		for x_ in xrange(-1,2):
			for y_ in xrange(-1,2):
				x_sum += dataArr[idx_x+x_,idx_y+y_]*(idx_x+x_)
				y_sum += dataArr[idx_x+x_,idx_y+y_]*(idx_y+y_)
		X = float(x_sum)/p_sum - 1
		Y = float(y_sum)/p_sum - 1
		if X < 0:
			X = 0
		if X > height:
			X = height
		if Y < 0:
			Y = 0
		if Y > width:
			Y = width

		return X/height, Y/width
	
	def coordinate2arr(self,coordinate, bottomArr):
		# calculate the mesh index of given coordinate (list)
		mesh_height = 1./bottomArr.shape[0]
		mesh_width = 1./bottomArr.shape[1]
		idx_x = int(coordinate[0]/mesh_height)
		idx_y = int(coordinate[1]/mesh_weight)
		
		cov = np.array([[0.3,0.],[0.,0.3]])
		#calculate the mesh value
		valueArr = np.zeros((mesh_height,mesh_width))
		for ii in xrange(idx_x-1,idx_x+2):
			for jj in xrange(idx_y-1,idx_y+2):
				tmp_sum = 0
				for xx_ in xrange(100):
					for yy_ in xrange(100):
						tmp_sum += self.pdf(np.array([ii+0.01*xx_*mesh_height,jj+0.01*yy_*mesh_width]),coordinate,cov)
				valueArr[ii,jj] = tmp_sum
		#normalize valueArr
		valueArr = valueArr/np.sum(valueArr)
		#normalize bottomArr
		bottom_min = np.min(bottomArr)
		bottom_sum = np.sum(bottomArr-bottom_min)

		valueArr = valueArr*bottom_sum + bottom_min
		return valueArr

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)
		for batch in xrange(bottom[0].data.shape[0]):
			for channel in xrange(bottom[0].data.shape[1]):
				x, y = self.arr2coordinate(bottom[0].data[batch,channel])
				top[0].data[batch,channel*2,1,1] = x
				top[0].data[batch,channel*2+1,1,1] = y
				
    def backward(self, bottom, top):
		for batch in xrange(bottom[0].data.shape[0]):
			for channel in xrange(bottom[0].data.shape[1]):
				for kk in xrange(7):
					bottom_blob = bottom[0].data[batch,channel,...]
					backward_bottom = self.coordinate2arr(top[0].data[batch,kk*2:kk*2+2,1,1],bottom_blob)
					bottom[0].diff[batch,channel,...] = back_bottom - bottom[0].data[batch,channel,...]
		

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
