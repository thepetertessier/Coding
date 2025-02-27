import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import time

# Audio parameters
RATE = 44100  # Sampling rate
CHUNK = 1024  # Buffer size per frame

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

# Function to convert frequency to musical note
def frequency_to_note(frequency):
    A4 = 440.0
    if frequency <= 0:
        return "Unknown"
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    n = round(12 * np.log2(frequency / A4))
    note = note_names[(n + 9) % 12]  # A4 is index 9
    octave = 4 + (n // 12)
    return f"{note}{octave}"


# Setup PyAudio for real-time input
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, 
                input=True, frames_per_buffer=CHUNK)
# Initialize plot
plt.ion()
fig, ax = plt.subplots()
x_data = []
y_data = []
line, = ax.plot([], [], '-o', label="Pitch (Hz)")
ax.set_xlim(0, 5)  # Time window of 5 seconds
ax.set_ylim(50, 500)  # Vocal range
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.legend()
start_time = time.time()

# Real-time loop
try:
    while plt.fignum_exists(fig.number):  # Check if figure is still open
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16).copy()
        pitch = estimate_pitch(data, RATE)
        current_time = time.time() - start_time
        
        if 50 < pitch < 500:  # Display within vocal range
            x_data.append(current_time)
            y_data.append(pitch)
            
            # Keep last 5 seconds of data
            x_data = [t for t in x_data if current_time - t <= 5]
            y_data = y_data[-len(x_data):]
            
            line.set_xdata(x_data)
            line.set_ydata(y_data)
            ax.set_xlim(max(0, current_time - 5), current_time)
            
            plt.draw()
            plt.pause(0.01)
        
        print(f"Detected Pitch: {pitch:.2f} Hz", flush=True)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    plt.close()
