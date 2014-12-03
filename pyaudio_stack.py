#ederwander
import pyaudio 
import numpy as np 
import wave 

chunk = 1024 
FORMAT = pyaudio.paInt16 
CHANNELS = 1 
RATE = 8800 
K=0 
DISTORTION = 0.61

p = pyaudio.PyAudio() 

stream = p.open(format = FORMAT, 
                channels = CHANNELS, 
                rate = RATE, 
                input = True, 
                output = True, 
                frames_per_buffer = chunk) 


print "Eng Eder de Souza - ederwander" 
print "Primitive Pedal" 


while(True): 

    data = stream.read(chunk) 
    data = np.fromstring(data, dtype=np.int16)  
    M = 2*DISTORTION/(1-DISTORTION);
    data = (1+M)*(data)/(1+K*abs(data));
    data = np.array(data, dtype='int16') 
    signal = wave.struct.pack("%dh"%(len(data)), *list(data))
    stream.write(signal) 

stream.stop_stream() 
stream.close() 
p.terminate() 