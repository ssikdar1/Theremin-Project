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

def main():	

	cap = cv2.VideoCapture(0)

	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()

		# Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Display the resulting frame
		cv2.imshow('frame',gray)
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