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

def adjustAmplitude(data):
	ampScale = .002
	data_array = numpy.fromstring(data, dtype='h')
	
	# Edit
	data_array = data_array * ampScale  # half amplitude
	data = pack('h'*len(data_array), *data_array)

	return data
		
CHUNK = 1024

if len(sys.argv) < 2:
    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)

wf = wave.open(sys.argv[1], 'rb')
data = wf.readframes(CHUNK)

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)


while data != '':
	scaledAmp = adjustAmplitude(data)	#scale the volume
	stream.write(scaledAmp)
	data = wf.readframes(CHUNK)

stream.stop_stream()
stream.close()

p.terminate()