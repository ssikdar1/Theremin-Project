#!/usr/bin/python

#for threading
import Queue
import threading
import time
import pyaudio
import numpy
import array
from math import sin,pi,log10
import traceback

import cv2
#get video from camera
cap = cv2.VideoCapture(0)
ret0, frame0 = cap.read()
cols, rows, dim = original_shape = tuple(frame0.shape)
intervalSize = rows/10

import random #remove later

#A class written to hold the audio functions
from audiolibrary import AudioLibrary
audiolib = AudioLibrary()

#A class written to hold the video functions
from videolibrary import VideoLibrary
videolib = VideoLibrary()

#flag to let threads no when they should run and die
#TODO better way to do this?
running = True

p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(2),
				channels=1,
				rate=44100,
				output=True)

music_buffer = Queue.Queue()

class ProducerThread(threading.Thread):
	def __init__(self):
		super(ProducerThread, self).__init__()
		#event flag initialized to true
		self.stoprequest = threading.Event()

	def run(self):
		nums = range(5)
		while not self.stoprequest.isSet():
			try:
				segment = audiolib.createA440(epsilon=0,note=random.choice(nums))
				# Capture frame-by-frame
				ret, frame = cap.read()
				halves = numpy.hsplit(frame,2)
				
				vol_contours, vol_hierarchy = cv2.findContours(numpy.copy( videolib.skinDetection(halves[0])),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
				vol_maxsize, vol_maxind, vol_boundrect = videolib.findLargestContour(vol_contours)
				#get and draw centroid
				vol_centerX, vol_centerY =  videolib.getCentroid(vol_contours, vol_maxind)
				scale = log10(10*log10(float(rows)/vol_centerY))
				ret = audiolib.adjustAmplitude(numpy.copy(input),scale-.4)
				music_buffer.put(segment)
			except Exception, e:
				print ("Producer")
				print str(e);
				continue
	def join(self, timeout=None):
		#overiding the join method, to be able to kill the thread if necessary
		self.stoprequest.set()
		super(ProducerThread,self).join(timeout)				

class ConsumerThread(threading.Thread):
	def __init__(self):
		super(ConsumerThread,self).__init__()
		self.stoprequest = threading.Event()

	def run(self):		
		while not self.stoprequest.isSet():
			try:
				# note that qsize is approximate
				size = music_buffer.qsize()
				if (size > 4):
					segment = music_buffer.get()
					stream.write(segment)
			except Exception, e:
				print str(e)
				traceback.print_exc()
				continue
	def join(self, timeout=None):
		#overiding the join method, to be able to kill the thread if necessary
		self.stoprequest.set()
		super(ProducerThread,self).join(timeout)

if __name__ == '__main__':
	
	print 'hello'
	running = True

	producerThread = ProducerThread()
	consumerThread = ConsumerThread()
	
	producerThread.start()
	consumerThread.start() 	
	
	while (running):
	
		try:
			continue
		except KeyboardInterrupt:
			print "Ctrl-c received! Sending kill to threads..."        
			producerThread.join()
			consumerThread.join()

