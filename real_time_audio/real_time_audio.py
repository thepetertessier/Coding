import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

# Audio parameters
RATE = 44100  # Sampling rate
CHUNK = 1024  # Buffer size per frame

# Setup PyAudio for real-time input
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, 
                input=True, frames_per_buffer=CHUNK)

# Plot setup
plt.ion()
fig, ax = plt.subplots()
x_data = np.arange(50, 500)  # Frequency range in Hz
y_data = np.zeros_like(x_data)
line, = ax.plot(x_data, y_data, 'r')
ax.set_ylim(0, 1)
ax.set_xlim(50, 500)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Autocorrelation Strength")
ax.set_title("Real-time Pitch Detection")

# FFT-based Autocorrelation function
def autocorrelate_fft(signal):
    N = len(signal)
    signal = signal.astype(np.float64)  # Convert to float64 before modifying
    signal -= np.mean(signal)  # Remove DC offset
    spectrum = fft(signal, n=2*N)  # Zero-padding
    power_spectrum = np.abs(spectrum) ** 2
    autocorr = ifft(power_spectrum).real[:N]  # Only first N values are valid
    return autocorr

# Function to estimate pitch from autocorrelation
def estimate_pitch(signal, rate):
    autocorr = autocorrelate_fft(signal)
    autocorr[:50] = 0  # Ignore small lags to remove DC components
    peak = np.argmax(autocorr)  # Find the first peak
    if peak == 0:
        return 0  # No valid pitch detected
    return rate / peak  # Convert lag to frequency

# Real-time loop
try:
    while True:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16).copy()
        pitch = estimate_pitch(data, RATE)
        if 50 < pitch < 500:  # Display within vocal range
            y_data = np.exp(-0.01 * (x_data - pitch) ** 2)  # Gaussian shape
            line.set_ydata(y_data)
        plt.pause(0.01)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    plt.close()
