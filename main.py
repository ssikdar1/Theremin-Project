#!/usr/bin/python
import Queue
import threading
import time
import pyaudio

#for video
import cv2
#get video from camera
cap = cv2.VideoCapture(0)

import random #remove later

from audiolibrary import AudioLibrary
audiolib = AudioLibrary()

p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(2),
				channels=1,
				rate=44100,
				output=True)

music_buffer = Queue.Queue()

def producer():
	""" 
		Producer Thread 
		Put a music segement onto the queue
	"""
	# Capture frame-by-frame
	ret, frame = cap.read()
	halves = numpy.hsplit(frame,2)
	
	nums = range(5)
	while True:
		segment = audiolib.createA440(epsilon=0,note=random.choice(nums))
		music_buffer.put(segment)

def consumer():
	""" consumer thread
		If the buffer is not empty
		pop an element and write it to the stream
	"""
	while True:
		try:
			# note that qsize is approximate
			size = music_buffer.qsize()
			if (size > 4):
				segment = music_buffer.get()
				stream.write(segment)
		except Exception, e:
			print "ERROR IN consumer"



if __name__ == '__main__':
	
	#print 'hello'
	#producerThread = threading.Thread(target=producer)
	#t.daemon = True
	#producerThread.start()

	#consumerThread = threading.Thread(target=consumer)
	#t.daemon = True
	#consumerThread.start()

	pass

