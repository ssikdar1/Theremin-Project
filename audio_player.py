#
# For cs585
#
#

#for streaming audio
import pyaudio	
import wave		
import sys
from struct import pack           # This creates the binary data structure for frames
import numpy
import array
from math import sin,pi

#for video
import cv2

def adjustAmplitude(data,ampScale):
	""" get incoming audio data and adjust the amplitude by a certain factor """
	data_array = numpy.fromstring(data, dtype='h')
	
	# Edit
	data_array = data_array * ampScale  # scale * amplitude
	data = pack('h'*len(data_array), *data_array)

	return data

def createA440():
	
	#c eb f f# g a Bb C
	notes = [261.63,311.13,349.23,269.99,392.0,440.0,466.16,523.25]
	numChannels = 1                      # mono
	sampleWidth = 2                      # in bytes, a 16-bit short
	sampleRate = 44100
	# MAX_AMP = 2**(8*sampleWidth - 1) - 1 #maximum amplitude is 2**15 - 1  = 32767 
	MAX_AMP = 2**(8*sampleWidth - 1) - 1 #maximum amplitude is 2**15 - 1  = 32767 
	lengthSeconds = .4                 
	numSamples = int(sampleRate * lengthSeconds)
	data = array.array("h")

	for i in range( numSamples ):
	 
		#  2 * pi * frequency is the angular velocity in radians/sec
		#  multiplying this by i / sampleRate incrementally creates angle at each sample
		#  and then sin ( angle ) => amplitude at this sample

		sample = (MAX_AMP) * sin( 2 * pi * 440.0 * i / sampleRate ) 

		data.append( int( sample ) )

	data_array = numpy.fromstring(data, dtype='h')
	# Edit
	data = pack('h'*len(data_array), *data_array)
	return data


def skinDetection(src):
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

	
def findLargestContour(contours):
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

def getCentroid(contours,maxind):
	#get the centroid which is the first moment
	moments = cv2.moments(contours[maxind])
	centroid_x = int(moments['m10']/moments['m00'])
	centroid_y = int(moments['m01']/moments['m00'])
	return centroid_x, centroid_y
	
def drawOptFlowMap(flow, cflowmap, step, color):
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
	
def main():	

	#if we wanted to get region
	#floor(float(point.y)/intervalSize)
	cap = cv2.VideoCapture(0)
	
	prevgray = None #used for optical flow
	
	ret0, frame0 = cap.read()
	cols, rows, dim = original_shape = tuple(frame0.shape)
	intervalSize = rows/10
	
	#
	# Audio
	#
	CHUNK = 1024
	p = pyaudio.PyAudio()
	stream = p.open(format=p.get_format_from_width(2),
					channels=1,
					rate=44100,
					output=True)

					
	ctr = 0
	input = createA440()
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		# reflectImage(frame)
		
		if(ctr%2==0):
			#line splitting the frame in half
			# cv2.line(frame,(320,0), (320,480), (255,255,255))
			
			halves = numpy.hsplit(frame,2)
			
			# # VOLUME HAND
			# #
			vol_skin = skinDetection(halves[0])
			
			vol_contours, vol_hierarchy = cv2.findContours(numpy.copy(vol_skin),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			vol_maxsize, vol_maxind, vol_boundrect = findLargestContour(vol_contours)
			
			# Draw contours
			vol_contour_output = numpy.zeros(halves[0].shape, dtype=numpy.uint8)
			# cv2.drawContours(vol_contour_output, vol_contours, vol_maxind, 	(255, 0, 0), cv2.cv.CV_FILLED, 8, vol_hierarchy)
			# cv2.drawContours(vol_contour_output, vol_contours, vol_maxind, (0,0,255), 2, 8, vol_hierarchy)
			
			#get and draw centroid
			vol_centerX, vol_centerY =  getCentroid(vol_contours, vol_maxind)
			# cv2.circle(vol_contour_output,(vol_centerX,vol_centerY),5,(255,255,255),8)
			cv2.circle(halves[0],(vol_centerX,vol_centerY),5,(255,255,255),8)
			
			#show all volume images
			cv2.imshow('volume raw',halves[0])
			# cv2.imshow('volume skin',vol_skin)
			# cv2.imshow('vol_contour_output',vol_contour_output)
					
			scale = 5**(float(vol_centerY)/1000.0)
			# print scale
			if 1.0/scale > 1:
				scale = 1
			print ('1/scale',1.0/scale)
			input = adjustAmplitude(input,1.0/scale)
			
			stream.write(input)
			
			
			######################################################
			######################################################
			#####################################################
			
			# #
			# # PITCH HAND
			# #
			

			# # OPTICAL FLOW
								
			# # Calculate Optical Flow for the left hand size
			# # call Farneback's optical flow algorithm
			# # Documentation: http://docs.opencv.org/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
			# # Link to Farneback's paper: http://www.diva-portal.org/smash/get/diva2:273847/FULLTEXT01.pdf
            # # calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);

			# gray = cv2.cvtColor(halves[1], cv2.COLOR_BGR2GRAY);
			# optFlowMap = gray
			
			# if(prevgray is not None):
				# #FRAME DIFFERENCING
				# diff = cv2.absdiff(gray,prevgray)
				
				# flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
				# # flow = cv2.calcOpticalFlowFarneback(prevgray, diff, 0.5, 3, 15, 3, 5, 1.2, 0)
				# cflow = cv2.cvtColor(prevgray, cv2.COLOR_GRAY2BGR);
				# optFlowMap = drawOptFlowMap(flow, cflow, 16, (0, 255, 0));
				

				# #get norm of matrix
				# # print flow.shape
				
				# threshold = .006
				# if(numpy.linalg.norm(flow[0], 1) > threshold or numpy.linalg.norm(flow[1], 1) > threshold):
					# print 'go yes hi'
					# print(numpy.linalg.norm(flow[0], 1),numpy.linalg.norm(flow[1], 1))
					# print ('sum',numpy.matrix(flow[0][0]))
			
			# # swap(prevgray, gray);
			# prevgray = gray
						
			
			
			# pitch_skin = skinDetection(halves[1])
			# # cv2.imshow('hand raw',halves[1])
			# # cv2.imshow('hand skin',pitch_skin)
			
			# pitch_contour_output = numpy.zeros(halves[1].shape, dtype=numpy.uint8)
			
			# # // Documentation for drawing rectangle: http://docs.opencv.org/modules/core/doc/drawing_functions.html
			# # try:
				# # cv2.rectangle(pitch_contour_output, (boundrec[0],boundrec[1]), (boundrec[0]  + boundrec[2], boundrec[1] + boundrec[3]),(0,255,0),1, 8,0);
				# # cv2.rectangle(pitch_contour_output, (boundrec2[0],boundrec2[1]), (boundrec2[0]  + boundrec2[2], boundrec2[1] + boundrec2[3]),(0,255,0),1, 8,0);
			# # except:
				# # pass
			
			# # cv2.circle(pitch_contour_output,(centroid_x,centroid_y),5,(255,255,255),8)
			# # cv2.circle(frame,(centroid_x,centroid_y),5,(255,255,255),8)

			
			# pitch_contours, pitch_hierarchy = cv2.findContours(pitch_skin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)		
			# pitch_maxsize = 0
			# pitch_maxind = 0
			# pitch_boundrec = None
			
			# for i in range(0, len(pitch_contours)):		
				# area = cv2.contourArea(pitch_contours[i]);
				# if (area > pitch_maxsize):
					# pitch_maxsize = area
					# pitch_maxind = i
					# pitch_boundrec = cv2.boundingRect(pitch_contours[i])

			# #Documentation for drawing contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours#drawcontours
			# cv2.drawContours(pitch_contour_output, pitch_contours, pitch_maxind, 	(255, 0, 0), cv2.cv.CV_FILLED, 8, pitch_hierarchy)
			# cv2.drawContours(pitch_contour_output, pitch_contours, pitch_maxind, (0,0,255), 2, 8, pitch_hierarchy)					
			
			# #get the centroid which is the first moment
			# pitch_moments = cv2.moments(pitch_contours[pitch_maxind])
			# pitch_centroid_x = int(pitch_moments['m10']/pitch_moments['m00'])
			# pitch_centroid_y = int(pitch_moments['m01']/pitch_moments['m00'])
			
			# # cv2.circle(frame,(centroid_x1+320,centroid_y1),5,(255,255,255),8)
			
			# cv2.imshow("pitch hand contour", pitch_contour_output)
			
			# cv2.circle(optFlowMap,(pitch_centroid_x,pitch_centroid_y),5,(255,255,255),8)
			# cv2.imshow("Optical Flow Map", optFlowMap);

	

		
		# lines spliting the right hand side into regions for the note values?
		# cv2.line(frame,(0,intervalSize), (320,intervalSize), (255,255,255))
		# cv2.line(frame,(0,intervalSize*2), (320,intervalSize*2), (255,255,255))
		# cv2.line(frame,(0,intervalSize*3), (320,intervalSize*3), (255,255,255))
		# cv2.line(frame,(0,intervalSize*4), (320,intervalSize*4), (255,255,255))
		# cv2.line(frame,(0,intervalSize*5), (320,intervalSize*5), (255,255,255))
		# cv2.line(frame,(0,intervalSize*6), (320,intervalSize*6), (255,255,255))
		# cv2.line(frame,(0,intervalSize*7), (320,intervalSize*7), (255,255,255))
		# cv2.line(frame,(0,intervalSize*8), (320,intervalSize*8), (255,255,255))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	#close the stream
	stream.stop_stream()
	stream.close()
	p.terminate()
	
main()