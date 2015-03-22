import sys
from struct import pack           # This creates the binary data structure for frames
import numpy
import array
from math import sin,pi,log10

class AudioLibrary:

	def __init__(self):
		pass

	def adjustAmplitude(self,data,ampScale):
		""" get incoming audio data and adjust the amplitude by a certain factor """
		data_array = numpy.fromstring(data, dtype='h')
	
		# Edit
		data_array = data_array * ampScale  # scale * amplitude
		data = pack('h'*len(data_array), *data_array)
		return data

	def createA440(self,epsilon, note):
		
		#c eb f f# g a Bb C
		notes = [261.63,311.13,349.23,269.99,392.0,440.0,466.16,523.25]
		numChannels = 1                      # mono
		sampleWidth = 2                      # in bytes, a 16-bit short
		# sampleRate = 44100
		sampleRate = 44100/4				#downsampling 
		# MAX_AMP = 2**(8*sampleWidth - 1) - 1 #maximum amplitude is 2**15 - 1  = 32767 
		MAX_AMP = 2**(8*sampleWidth - 1) - 1 #maximum amplitude is 2**15 - 1  = 32767 
		lengthSeconds = 2           
		numSamples = int(sampleRate * lengthSeconds)
		data = array.array("h")

		for i in range( numSamples ):
		 
			#  2 * pi * frequency is the angular velocity in radians/sec
			#  multiplying this by i / sampleRate incrementally creates angle at each sample
			#  and then sin ( angle ) => amplitude at this sample
			"""
			if(i%2 == 0):
				sample = (MAX_AMP) * sin( 2 * pi * (440.0) * i / sampleRate ) 
			else:
				sample = (MAX_AMP) * sin( 2 * pi * (440.0 + epsilon) * i / sampleRate ) 
			"""
			
			LFO = sin(2 * pi * .5 * i / sampleRate)
			sample = (MAX_AMP) * sin((2 * pi * (notes[note]) * i / sampleRate) + (epsilon/500) * LFO)
			#sample = (MAX_AMP) * sin(2 * pi * (440.0 ) * i / sampleRate)
			
			data.append( int( sample ) )

		data_array = numpy.fromstring(data, dtype='h')
		# Edit
		data = pack('h'*len(data_array), *data_array)
		return data