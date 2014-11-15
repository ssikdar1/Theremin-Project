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

def reflectImage(src):
	"""
	For my own sanity so my movements are not reversed when looking at a camera
	"""	
	# dst.create( image.size(), image.type() );
	# map_x.create( image.size(), CV_32FC1 );
	# map_y.create( image.size(), CV_32FC1 );
		# for( int j = 0; j < image.rows; j++ )
	# { 
		# for( int i = 0; i < image.cols; i++ )
		# {
			# map_x.at<float>(j,i) = image.cols - i ;
			# map_y.at<float>(j,i) = j ;
		# }
	# }
	cols, rows, dim = original_shape = tuple(src.shape)
	map_x = numpy.empty( (cols,rows,1));
	map_y = numpy.empty((cols,rows,1));

	for j in range(0,rows):
		for i in range(0,cols):
			map_x.itemset((i,j,0),j)
			map_y.itemset((i,j,0),i)

	map_x_32 = map_x.astype('float32')
	map_y_32 = map_y.astype('float32')
	
	dst = cv2.remap( src, map_x_32, map_y_32, cv2.cv.CV_INTER_LINEAR);
	cv2.imshow('dst',dst)
	cv2.waitKey(5)

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
	
def main():	

	#if we wanted to get region
	#floor(float(point.y)/intervalSize)
	cap = cv2.VideoCapture(0)
	
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		# reflectImage(frame)
		ctr = 0
		
		if(ctr%2==0):
			cv2.line(frame,(320,0), (320,480), (255,255,255))
			halves = numpy.hsplit(frame,2)
			
			skin = skinDetection(halves[0])
			contours, hierarchy = cv2.findContours(skin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			
			maxsize = 0
			maxind = 0
			maxsize2 = 0
			maxind2 = 0
			boundrec = None
			boundrec2 = None
			
			for i in range(0, len(contours)):		
				area = cv2.contourArea(contours[i]);
				if (area > maxsize):
					boundrec2 = boundrec
					maxsize = area
					maxind = i
					boundrec = cv2.boundingRect(contours[i])

			# Draw contours
			contour_output = numpy.zeros(halves[0].shape, dtype=numpy.uint8)
			#Documentation for drawing contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours#drawcontours
			cv2.drawContours(contour_output, contours, maxind, 	(255, 0, 0), cv2.cv.CV_FILLED, 8, hierarchy)
			cv2.drawContours(contour_output, contours, maxind, (0,0,255), 2, 8, hierarchy)
			
			#get the centroid which is the first moment
			moments = cv2.moments(contours[maxind])
			centroid_x = int(moments['m10']/moments['m00'])
			centroid_y = int(moments['m01']/moments['m00'])
			print str(centroid_x) + " " + str(centroid_y)
			
			# // Documentation for drawing rectangle: http://docs.opencv.org/modules/core/doc/drawing_functions.html
			try:
				cv2.rectangle(contour_output, (boundrec[0],boundrec[1]), (boundrec[0]  + boundrec[2], boundrec[1] + boundrec[3]),(0,255,0),1, 8,0);
				cv2.rectangle(contour_output, (boundrec2[0],boundrec2[1]), (boundrec2[0]  + boundrec2[2], boundrec2[1] + boundrec2[3]),(0,255,0),1, 8,0);
				
			except:
				pass
			
			cv2.circle(contour_output,(centroid_x,centroid_y),5,(255,255,255),8)

		# # Our operations on the frame come here
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# Display the resulting frame
			# cv2.imshow('h1 right', halves[0])
			cv2.imshow('h2', halves[1])
			
			cv2.imshow('raw',frame)
			cv2.imshow('frame right',contour_output)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


	
main()