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
	"""
	cols, rows, dim = original_shape = tuple(src.shape)
	dst = numpy.zeros(original_shape)
	for i in range(0,rows):
		for j in range(0,cols):
			B = src.item(j,i,0)
			G = src.item(j,i,1)
			R = src.item(j,i,2)
		
			if(R > 95 and G > 40 and B > 20 and max(R,G,B)-min(R,G,B) > 15 and abs(R-G) > 15 and R > G and R > B):
				dst.itemset((j,i,0),255)
				dst.itemset((j,i,1),255)
				dst.itemset((j,i,2),255)
	return dst

def main():	

	cap = cv2.VideoCapture(0)

	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		skin = skinDetection(frame)
		contours, hierarchy = cv2.findContours(skin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		for h,cnt in enumerate(contours):
			mask = numpy.zeros(imgray.shape,np.uint8)
		cv2.drawContours(mask,[cnt],0,255,-1)
		mean = cv2.mean(im,mask = mask)
		
		# # Our operations on the frame come here
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# Display the resulting frame
		cv2.imshow('frame',skin)
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