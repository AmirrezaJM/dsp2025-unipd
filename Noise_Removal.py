import numpy as np  
import librosa 
import soundfile as sf 
import scipy.signal as signal  
import matplotlib.pyplot as plt

def load_audio_file(file_path):
    audio_signal, sample_rate = librosa.load(file_path, sr=None)
    return audio_signal, sample_rate

def add_noise(audio, noise_factor=0.05):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def reduce_noise_spectral_filter(audio_signal, fft_size=2048, hop_step=512):
    spectrum = librosa.stft(audio_signal, n_fft=fft_size, hop_length=hop_step)
    magnitude, _ = librosa.magphase(spectrum)
    estimated_noise = np.mean(magnitude, axis=1, keepdims=True)
    filtered_magnitude = np.maximum(magnitude - estimated_noise, 0.2 * magnitude)
    filtered_spectrum = filtered_magnitude * np.exp(1j * np.angle(spectrum))
    filtered_audio = librosa.istft(filtered_spectrum, hop_length=hop_step)
    plot_filtered_audio(audio_signal, filtered_audio, "Spectral Subtraction")
    return filtered_audio

def reduce_noise_wiener_filter(audio_signal, fft_window=2048, hop_size=512):
    frequency_spectrum = librosa.stft(audio_signal, n_fft=fft_window, hop_length=hop_size)
    magnitude_spectrum, _ = librosa.magphase(frequency_spectrum)
    estimated_noise_level = np.mean(magnitude_spectrum, axis=1, keepdims=True)
    wiener_gain = magnitude_spectrum**2 / (magnitude_spectrum**2 + estimated_noise_level**2)
    refined_magnitude = magnitude_spectrum * wiener_gain
    refined_spectrum = refined_magnitude * np.exp(1j * np.angle(frequency_spectrum))
    filtered_audio = librosa.istft(refined_spectrum, hop_length=hop_size)

    plot_filtered_audio(audio_signal, filtered_audio, "Wiener Filtering")

    return filtered_audio


def reduce_noise_adaptive(audio_signal, learning_rate=0.01, filter_size=32):
    total_samples = len(audio_signal)
    enhanced_audio = np.zeros(total_samples)
    filter_weights = np.zeros(filter_size)
    
    for i in range(filter_size, total_samples):
        reference_chunk = audio_signal[i-filter_size:i]    
        if len(reference_chunk) != filter_size:
            continue
        predicted_noise = np.dot(filter_weights, reference_chunk)
        error_signal = audio_signal[i] - predicted_noise
        filter_weights += 2 * learning_rate * error_signal * reference_chunk
        enhanced_audio[i] = predicted_noise
    
    plot_filtered_audio(audio_signal, enhanced_audio, "Adaptive Filtering")
    return enhanced_audio

def reduce_noise_chebyshev_filter(audio, sr, low_cutoff=100, high_cutoff=1100, order=2, ripple=0.4):
    nyquist = 0.5 * sr
    b, a = signal.cheby1(order, ripple, [low_cutoff / nyquist, high_cutoff / nyquist], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio)
    
    plot_filtered_audio(audio, filtered_audio, "Chebyshev Filtering")
    return filtered_audio

def calculate_snr(clean, noisy):
    min_length = min(len(clean), len(noisy))
    clean = clean[:min_length]
    noisy = noisy[:min_length]
    noise = noisy - clean
    sum_clean = np.sum(clean ** 2);
    sum_noise = np.sum(noise ** 2)
    return 10 * np.log10(sum_clean / sum_noise)

def plot_filtered_audio(original, filtered, method_name):
    plt.figure(figsize=(12, 5))
    plt.plot(original, label="Original Signal", alpha=0.7)
    plt.plot(filtered, label=f"{method_name} Output", alpha=0.7)
    plt.title(f"{method_name} Comparison")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

def process_audio(input_file, method, output_file):
    audio_signal, sample_rate = load_audio_file(input_file)
    noisy_audio = add_noise(audio_signal)
    
    if method == "spectral_subtraction":
        enhanced_audio = reduce_noise_spectral_filter(noisy_audio)
    elif method == "wiener_filtering":
        enhanced_audio = reduce_noise_wiener_filter(noisy_audio)
    elif method == "chebyshev_filtering":
        enhanced_audio = reduce_noise_chebyshev_filter(noisy_audio, sample_rate)
    elif method == "adaptive_filtering":
        enhanced_audio = reduce_noise_adaptive(noisy_audio)
    else:
        raise ValueError("Invalid method.")
    
    sf.write(output_file, enhanced_audio, sample_rate)
    
    snr_before = calculate_snr(audio_signal, noisy_audio)
    snr_after = calculate_snr(audio_signal, enhanced_audio)
    print(f"SNR before: {snr_before:.2f} dB, SNR after {method}: {snr_after:.2f} dB")
    
    return enhanced_audio, sample_rate

if __name__ == "__main__":
    # Define noise reduction techniques and corresponding output filenames
    noise_reduction_methods = [
        "spectral_subtraction",
        "wiener_filtering",
        "chebyshev_filtering",
        "adaptive_filtering"
    ]
    
    output_filenames = [
        "enhanced_spectral.wav",
        "enhanced_wiener.wav",
        "enhanced_chebyshev.wav",
        "enhanced_adaptive.wav"
    ]

    # Apply each noise reduction method to the input audio file
    for method, output_file in zip(noise_reduction_methods, output_filenames):
        process_audio("sample1.wav", method, output_file)
