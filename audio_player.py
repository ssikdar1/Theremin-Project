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
	data_array = data_array * ampScale  # half amplitude
	data = pack('h'*len(data_array), *data_array)

	return data

def createA440():
	
	#c eb f f# g a Bb C
	notes = [261.63,311.13,349.23,269.99,392.0,440.0,466.16,523.25]
	numChannels = 1                      # mono
	sampleWidth = 2                      # in bytes, a 16-bit short
	sampleRate = 44100
	MAX_AMP = 2**(8*sampleWidth - 1) - 1 #maximum amplitude is 2**15 - 1  = 32767 
	lengthSeconds = .4                 
	numSamples = int(sampleRate * lengthSeconds)
	data = array.array("h")

	for i in range( numSamples ):
	 
		#  2 * pi * frequency is the angular velocity in radians/sec
		#  multiplying this by i / sampleRate incrementally creates angle at each sample
		#  and then sin ( angle ) => amplitude at this sample

		sample = MAX_AMP * sin( 2 * pi * 440.0 * i / sampleRate ) 

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

def audio(input):
	# Eventually put the audio stream in the while loop
	CHUNK = 1024
	# data = input.readframes(CHUNK)
	# data = input.readframes(CHUNK)

	p = pyaudio.PyAudio()

	stream = p.open(format=p.get_format_from_width(2),
					channels=1,
					rate=44100,
					output=True)

	stream.write(input)
	# while input != '':
		# # scaledAmp = adjustAmplitude(data,1)	#scale the volume
		# stream.write(input)
		# data = input.readframes(CHUNK)

	stream.stop_stream()
	stream.close()

	p.terminate()
	
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
	
def main():	

	#if we wanted to get region
	#floor(float(point.y)/intervalSize)
	cap = cv2.VideoCapture(0)
	
	prevgray = None #used for optical flow
	ret0, frame0 = cap.read()
	cols, rows, dim = original_shape = tuple(frame0.shape)
	intervalSize = rows/10

	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		# reflectImage(frame)
		ctr = 0
		
		if(ctr%2==0):
			#line splitting the frame in half
			cv2.line(frame,(320,0), (320,480), (255,255,255))
			

			
			halves = numpy.hsplit(frame,2)
			cols, rows, dim = original_shape = tuple(frame.shape)
			
			vol_skin = skinDetection(halves[0])
			
			vol_contours, vol_hierarchy = cv2.findContours(numpy.copy(vol_skin),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			vol_maxsize, vol_maxind, vol_boundrect = findLargestContour(vol_contours)
			
			# Draw contours
			vol_contour_output = numpy.zeros(halves[0].shape, dtype=numpy.uint8)
			cv2.drawContours(vol_contour_output, vol_contours, vol_maxind, 	(255, 0, 0), cv2.cv.CV_FILLED, 8, vol_hierarchy)
			cv2.drawContours(vol_contour_output, vol_contours, vol_maxind, (0,0,255), 2, 8, vol_hierarchy)
			
			#get and draw centroid
			# vol_centerX, vol_centerY =  getCentroid(vol_contours, vol_maxind)
			# print str(vol_centerX) + " " + str(vol_centerY)
			# cv2.circle(vol_contour_output,(vol_centerX,vol_centerY),5,(255,255,255),8)
			# cv2.circle(halves[0],(vol_centerX,vol_centerY),5,(255,255,255),8)
			
			#show all volume images
			cv2.imshow('volume raw',halves[0])
			cv2.imshow('volume skin',vol_skin)
			cv2.imshow('vol_contour_output',vol_contour_output)
			
			
			# pitch_skin = skinDetection(halves[1])
			# cv2.imshow('hand raw',halves[1])
			# cv2.imshow('hand skin',pitch_skin)
			
			
			# # // Documentation for drawing rectangle: http://docs.opencv.org/modules/core/doc/drawing_functions.html
			# try:
				# cv2.rectangle(contour_output, (boundrec[0],boundrec[1]), (boundrec[0]  + boundrec[2], boundrec[1] + boundrec[3]),(0,255,0),1, 8,0);
				# cv2.rectangle(contour_output, (boundrec2[0],boundrec2[1]), (boundrec2[0]  + boundrec2[2], boundrec2[1] + boundrec2[3]),(0,255,0),1, 8,0);
			# except:
				# pass
			
			# cv2.circle(contour_output,(centroid_x,centroid_y),5,(255,255,255),8)
			# cv2.circle(frame,(centroid_x,centroid_y),5,(255,255,255),8)

			# #
			# # get Centroid of the Pitch Hand
			# #
			# skin2 = skinDetection(halves[1])
			
			# contours1, hierarchy1 = cv2.findContours(skin2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)		
			# maxsize1 = 0
			# maxind1 = 0
			# boundrec1 = None
			
			# for i in range(0, len(contours1)):		
				# area = cv2.contourArea(contours1[i]);
				# if (area > maxsize1):
					# maxsize = area
					# maxind1 = i
					# boundrec1 = cv2.boundingRect(contours1[i])

			# # Draw contours
			# contour_output1 = numpy.zeros(halves[1].shape, dtype=numpy.uint8)
			# #Documentation for drawing contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours#drawcontours
			# cv2.drawContours(contour_output1, contours1, maxind1, 	(255, 0, 0), cv2.cv.CV_FILLED, 8, hierarchy1)
			# cv2.drawContours(contour_output1, contours1, maxind1, (0,0,255), 2, 8, hierarchy1)					
			
			# #get the centroid which is the first moment
			# moments1 = cv2.moments(contours1[maxind1])
			# centroid_x1 = int(moments1['m10']/moments1['m00'])
			# centroid_y1 = int(moments1['m01']/moments1['m00'])
			
			# cv2.circle(frame,(centroid_x1+320,centroid_y1),5,(255,255,255),8)
			
			# cv2.imshow("volume hand", contour_output1)
			# cv2.imshow("volume hand skin", skin2)
			# cv2.imshow("volume hand contours ", halves[1])
			
			
			#
			# Calculate Optical Flow for the left hand size
			#call Farneback's optical flow algorithm
			#Documentation: http://docs.opencv.org/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
			#Link to Farneback's paper: http://www.diva-portal.org/smash/get/diva2:273847/FULLTEXT01.pdf
            # calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);

			# gray = cv2.cvtColor(halves[1], cv2.COLOR_BGR2GRAY);
			
			# if(prevgray is not None):
				# uflow = cv2.calcOpticalFlowFarneback(prev, next, 0.5, 3, 15, 3, 5, 1.2, 0)
				# # cv2.cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
				# # drawOptFlowMap(uflow, cflow, 16, Scalar(0, 255, 0));
				# # imshow("Optical Flow", cflow);
        	
			# # swap(prevgray, gray);
			# prevgray = gray
			
			
			# Display the resulting frame
			# cv2.imshow('h1 right', halves[0])
			# cv2.imshow('h2', halves[1])
			
			# cv2.imshow('raw',frame)
			# cv2.imshow('frame right',contour_output)
		lines spliting the right hand side into regions for the note values?
		cv2.line(frame,(0,intervalSize), (320,intervalSize), (255,255,255))
		cv2.line(frame,(0,intervalSize*2), (320,intervalSize*2), (255,255,255))
		cv2.line(frame,(0,intervalSize*3), (320,intervalSize*3), (255,255,255))
		cv2.line(frame,(0,intervalSize*4), (320,intervalSize*4), (255,255,255))
		cv2.line(frame,(0,intervalSize*5), (320,intervalSize*5), (255,255,255))
		cv2.line(frame,(0,intervalSize*6), (320,intervalSize*6), (255,255,255))
		cv2.line(frame,(0,intervalSize*7), (320,intervalSize*7), (255,255,255))
		cv2.line(frame,(0,intervalSize*8), (320,intervalSize*8), (255,255,255))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


	
main()