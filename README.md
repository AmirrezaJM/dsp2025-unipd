# **Noise Removal - Audio Processing**

## **Overview**
This project implements multiple **noise reduction techniques** for audio processing using Python for DSP project in 2024-2025 with professor Battisti. The script takes an input audio file, adds synthetic noise, applies a chosen noise reduction method, and saves the enhanced output.

## **Functions**
- **Load and Process Audio**: Reads audio files and prepares them for processing.
- **Add Noise**: Simulates a noisy environment by injecting Gaussian noise.
- **Noise Reduction Techniques**:
  - **Spectral Subtraction** (`reduce_noise_spectral_filter`)
  - **Wiener Filtering** (`reduce_noise_wiener_filter`)
  - **Adaptive Filtering** (`reduce_noise_adaptive`)
  - **Chebyshev Filter** (`reduce_noise_chebyshev_filter`)
- **Signal-to-Noise Ratio (SNR) Calculation**: Measures noise reduction effectiveness.
- **Visualization**: Plots and compares original vs. filtered signals.

## **Installation**
### **1. Install Dependencies**
Ensure you have Python installed, then install required libraries:
```sh
pip install numpy librosa soundfile scipy matplotlib
```
for the some mac os you have to use a virtual environement.
run this code here 
```sh 
    python3 -m venv path/to/venv
    source path/to/venv/bin/activate
    python3 -m pip install ...
```
OR you can install libraries with Homebrew to install globally

## **Usage**
### **1. Running the Script**
Execute the script to process by this command:
```sh
python Noise_Removal.py
```
for the some mac os you have to write python3 execute the script.


## **Example Output**
The script generates the following enhanced audio files:
- `enhanced_spectral.wav`
- `enhanced_wiener.wav`
- `enhanced_chebyshev.wav`
- `enhanced_adaptive.wav`

Additionally, it prints the **SNR before and after processing** to show the effectiveness of each method.

## **Contributing**
Feel free to contribute by improving filtering techniques, adding **deep learning-based noise suppression**, or enhancing visualization methods.

## **License**
This project is open-source and available under the **MIT License**.

