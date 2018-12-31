from __future__ import division

import glob
import scipy.signal as sig
import soundfile as sf
import numpy as np
import sys


def read_file(file):
    signal, w = sf.read(file)
    if len(signal.shape) > 1:
        signal = [s[0] for s in signal]
    return signal, w


def decimate(signal, w):
    samples = len(signal)

    # t = probki/ f probkowania
    audio_duration = float(samples) / w

    # funkcja okna
    # signal = signal * np.kaiser(samples, 7)
    # signal = signal * np.hamming(samples)
    signal = signal * np.hanning(samples)

    spectrum = np.log(abs(np.fft.fft(signal)))
    # spectrum = np.fft.fft(signal)
    spec_dec = spectrum.copy()

    # funkcja decimate
    for beta in [2, 3, 4, 5]:
        decimated = sig.decimate(spectrum, beta)
        spec_dec[:len(decimated)] *= decimated
    peak_start = round(50 * audio_duration)

    # argumentu dla najwiekszej wartosci
    peak = np.argmax(spec_dec[peak_start:])
    f = (peak_start + peak) / audio_duration
    return f

# The voiced speech of a typical adult male will have a fundamental frequency from 85 to 180 Hz,
# and that of a typical adult female from 165 to 255 Hz.
def check(file, accuracy):
    signal, w = read_file(file)
    f = decimate(signal, w)
    if f <= 172.5:
        gender = 'M'
    elif 172.5 < f:
        gender = 'K'
    else:
        gender = 'M'
    value = file.replace(".wav", "")[-1:]
    if gender == value:
        accuracy += 1
    print(value, f, gender)
    return accuracy


if __name__ == "__main__":
    # files = glob.glob(sys.argv[1] + "/*.wav")
    files = glob.glob("./trainall/*.wav")
    accuracy = 0
    for file in files:
        accuracy = check(file, accuracy)
        print(accuracy)