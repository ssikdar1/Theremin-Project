import cv2
import numpy 

class VideoLibrary:
	def skinDetection(self,src):
		""" TODO optimize 
			http://opencvpython.blogspot.com/2012/06/fast-array-manipulation-in-numpy.html
			-Also fix this so its returning a grayscale
		"""
		cols, rows, dim = original_shape = tuple(src.shape)
		dst = numpy.zeros((cols,rows,1), numpy.uint8)
		for i in range(0,rows):
			for j in range(0,cols):
				B = src.item(j,i,0)
				G = src.item(j,i,1)
				R = src.item(j,i,2)
			
				if(R > 95 and G > 40 and B > 20 and max(R,G,B)-min(R,G,B) > 15 and abs(R-G) > 15 and R > G and R > B):
					dst.itemset((j,i,0),255)
					# dst.itemset((j,i,1),255)
					# dst.itemset((j,i,2),255)
				else:
					dst.itemset((j,i,0),0)
					# dst.itemset((j,i,1),0)
					# dst.itemset((j,i,2),0)
		return dst

		
	def findLargestContour(self,contours):
		maxsize = 0
		maxind = 0
		boundrec = None
		
		for i in range(0, len(contours)):		
			area = cv2.contourArea(contours[i]);
			if (area > maxsize):
				boundrec2 = boundrec
				maxsize = area
				maxind = i
				boundrec = cv2.boundingRect(contours[i])
		return maxsize,maxind,boundrec

	def getCentroid(self,contours,maxind):
		#get the centroid which is the first moment
		moments = cv2.moments(contours[maxind])
		centroid_x = int(moments['m10']/moments['m00'])
		centroid_y = int(moments['m01']/moments['m00'])
		return centroid_x, centroid_y
		
	def drawOptFlowMap(self,flow, cflowmap, step, color):
		y = 0
		x = 0
		cols, rows, dim = original_shape = tuple(cflowmap.shape)
		while(y < rows):
			while( x < cols):
				#fxy = (flow[y][x][0],flow[y][x][1])
				fxy = (flow.item(x,y,0), flow.item(x,y,1))
				cv2.line(cflowmap,(y,x), (cv2.cv.Round(fxy[1] + y),cv2.cv.Round(fxy[0] + x)), color)
				cv2.circle(cflowmap,(y,x),2,color,-1)
				x += step
			#print('y',y)
			y += step
			x = 0
		return cflowmap