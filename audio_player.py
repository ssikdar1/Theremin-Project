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

#for video
import cv2

def adjustAmplitude(data,ampScale):
	""" get incoming audio data and adjust the amplitude by a certain factor """
	data_array = numpy.fromstring(data, dtype='h')
	
	# Edit
	data_array = data_array * ampScale  # half amplitude
	data = pack('h'*len(data_array), *data_array)

	return data

def skinDetection(src):
	""" TODO optimize 
		http://opencvpython.blogspot.com/2012/06/fast-array-manipulation-in-numpy.html
		-Also fix this so its returning a grayscale
	"""
	cols, rows, dim = original_shape = tuple(src.shape)
	dst = src
	for i in range(0,rows):
		for j in range(0,cols):
			B = src.item(j,i,0)
			G = src.item(j,i,1)
			R = src.item(j,i,2)
		
			if(R > 95 and G > 40 and B > 20 and max(R,G,B)-min(R,G,B) > 15 and abs(R-G) > 15 and R > G and R > B):
				dst.itemset((j,i,0),255)
				dst.itemset((j,i,1),255)
				dst.itemset((j,i,2),255)
			else:
				dst.itemset((j,i,0),0)
				dst.itemset((j,i,1),0)
				dst.itemset((j,i,2),0)
	return dst

def main():	

	cap = cv2.VideoCapture(0)

	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		ctr = 0
		
		if(ctr%1==0):
			skin = skinDetection(frame)
			binSkin = cv2.cvtColor(skin,cv2.cv.CV_RGB2GRAY);
			contours, hierarchy = cv2.findContours(binSkin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			
			maxsize = 0
			maxind = 0
			maxsize2 = 0
			maxind2 = 0
			boundrec = None
			boundrec2 = None
			
			for i in range(0, len(contours)):		
				area = cv2.contourArea(contours[i]);
				if (area > maxsize):
					maxsize2 = maxsize
					maxind2 = maxind
					boundrec2 = boundrec
					maxsize = area
					maxind = i
					boundrec = cv2.boundingRect(contours[i])
				else:
					if(area > maxsize2):
						maxsize2 = area
						maxind = i
						boundrec2 = cv2.boundingRect(contours[i])
			# Draw contours
			contour_output = numpy.zeros(frame.shape, dtype=numpy.uint8)
			#Documentation for drawing contours: http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours#drawcontours
			cv2.drawContours(contour_output, contours, maxind, 	(255, 0, 0), cv2.cv.CV_FILLED, 8, hierarchy)
			cv2.drawContours(contour_output, contours, maxind, (0,0,255), 2, 8, hierarchy)
			cv2.drawContours(contour_output, contours, maxind2, (255, 0, 0), cv2.cv.CV_FILLED, 8, hierarchy)
			cv2.drawContours(contour_output, contours, maxind2, (0,0,255), 2, 8, hierarchy)
			# // Documentation for drawing rectangle: http://docs.opencv.org/modules/core/doc/drawing_functions.html
			cv2.rectangle(contour_output, (boundrec[0],boundrec[1]), (boundrec[0]  + boundrec[2], boundrec[1] + boundrec[3]),(0,255,0),1, 8,0);
			cv2.rectangle(contour_output, (boundrec2[0],boundrec2[1]), (boundrec2[0]  + boundrec2[2], boundrec2[1] + boundrec2[3]),(0,255,0),1, 8,0);
				

		
		# # Our operations on the frame come here
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# Display the resulting frame
			cv2.imshow('frame',contour_output)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

# Eventually put the audio stream in the while loop
# CHUNK = 1024

	# if len(sys.argv) < 2:
		# print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
		# sys.exit(-1)

	# wf = wave.open(sys.argv[1], 'rb')
	# data = wf.readframes(CHUNK)

	# p = pyaudio.PyAudio()

	# stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
					# channels=wf.getnchannels(),
					# rate=wf.getframerate(),
					# output=True)


	# while data != '':
		# scaledAmp = adjustAmplitude(data)	#scale the volume
		# stream.write(scaledAmp)
		# data = wf.readframes(CHUNK)

	# stream.stop_stream()
	# stream.close()

	# p.terminate()
	
main()