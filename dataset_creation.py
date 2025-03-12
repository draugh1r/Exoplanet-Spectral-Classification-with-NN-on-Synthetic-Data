import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Increased complexity
num_samples = 100000  # 2x more samples
num_features = 200    # 2x spectral resolution
num_classes = 5

planet_types = ["Earth-like", "Venus-like", "Hot Jupiter", "Mini-Neptune", "Exotic"]

def generate_telluric_contamination(x):
    """Simulate Earth's atmospheric contamination"""
    telluric = np.zeros_like(x)
    # Water vapor bands
    for peak in [0.7, 0.9, 1.1, 1.4, 1.9]:
        width = 0.1 * (1 + 0.2 * np.random.randn())
        depth = 0.3 * np.random.rand()
        telluric += depth * np.exp(-((x - peak) ** 2) / (2 * width**2))
    return telluric

def generate_instrument_response(x):
    """Simulate realistic instrument response"""
    response = 1.0 + 0.2 * np.sin(2 * np.pi * x / 0.5)
    response *= (1 + 0.1 * np.random.randn(len(x)))
    return savgol_filter(response, 11, 3)

def generate_spectrum_and_label(class_idx, x):
    """Generate more complex spectra with realistic effects"""
    
    # Temperature variations based on orbital position
    orbital_phase = np.random.rand() * 2 * np.pi
    base_temps = {
        0: (288, 20),    # Earth-like
        1: (737, 30),    # Venus-like
        2: (1500, 300),  # Hot Jupiter
        3: (400, 50),    # Mini-Neptune
        4: (600, 100)    # Exotic
    }
    
    base_temp, temp_var = base_temps[class_idx]
    temp = base_temp + temp_var * np.sin(orbital_phase)
    
    # Baseline continuum with temperature dependence
    continuum = np.exp(-((x - 1.5) ** 2) / (2 * (0.8 + 0.2 * np.random.rand())**2))
    continuum *= (temp / 288) ** 0.25  # Black body approximation
    
    # More complex absorption features with temperature broadening
    absorption_features = {
        0: [(0.76, 0.1, 0.4), (1.0, 0.15, 0.5), (1.4, 0.12, 0.6),
            (0.69, 0.08, 0.3), (1.27, 0.11, 0.45)],  # Earth-like
        1: [(1.1, 0.12, 0.5), (1.6, 0.14, 0.4), (2.0, 0.16, 0.6),
            (0.85, 0.1, 0.35), (1.85, 0.13, 0.55)],  # Venus-like
        2: [(0.6, 0.12, 0.5), (0.9, 0.14, 0.5), (1.3, 0.15, 0.6),
            (0.77, 0.11, 0.4), (1.55, 0.13, 0.45)],  # Hot Jupiter
        3: [(0.7, 0.13, 0.5), (1.2, 0.14, 0.5), (1.8, 0.16, 0.6),
            (0.95, 0.12, 0.45), (1.65, 0.15, 0.5)],  # Mini-Neptune
        4: [(0.5, 0.12, 0.6), (1.0, 0.15, 0.5), (1.7, 0.14, 0.5),
            (0.8, 0.11, 0.4), (1.45, 0.13, 0.55)]    # Exotic
    }

    # Add some random features from other classes (atmospheric mixing)
    features_for_class = absorption_features[class_idx].copy()  # Create a copy of original features
    if np.random.rand() < 0.3:  # 30% chance of mixed atmospheres
        other_class = np.random.choice([i for i in range(num_classes) if i != class_idx])
        other_features = np.array(absorption_features[other_class])
        selected_features = other_features[np.random.choice(len(other_features), size=2, replace=False)]
        features_for_class.extend(selected_features.tolist())

    spectrum = continuum.copy()
    
    # Add absorption features with temperature-dependent broadening
    for peak, base_width, depth_factor in features_for_class:
        # Temperature affects line width (thermal broadening)
        width = base_width * np.sqrt(temp / 288)
        # Random variations in depth
        depth = depth_factor * (0.5 + 0.8 * np.random.rand())
        
        # Asymmetric absorption features
        asymmetry = 0.4 * np.random.rand() - 0.2
        left_wing = -((x[x <= peak] - peak) ** 2) / (2 * (width * (1 + asymmetry))**2)
        right_wing = -((x[x > peak] - peak) ** 2) / (2 * (width * (1 - asymmetry))**2)
        absorption = np.concatenate([np.exp(left_wing), np.exp(right_wing)])
        
        spectrum -= depth * absorption

    # Add telluric contamination
    if np.random.rand() < 0.8:  # 80% chance of telluric contamination
        spectrum -= 0.7 * generate_telluric_contamination(x)

    # Add instrument response
    spectrum *= generate_instrument_response(x)

    # Add complex noise
    # White noise (detector)
    white_noise = np.random.normal(0, 0.02 + 0.03 * np.abs(np.sin(2 * np.pi * x / 0.5)), num_features)
    
    # Red noise (systematic trends)
    red_noise = np.zeros(num_features)
    for i in range(1, num_features):
        red_noise[i] = 0.8 * red_noise[i-1] + 0.2 * np.random.normal(0, 0.02)
    
    # Fringing noise (interference effects)
    fringing = 0.05 * np.sin(20 * x + 0.5 * np.random.rand())
    
    spectrum += white_noise + red_noise + fringing

    # Normalize but preserve some scale information
    spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
    spectrum *= 0.7 + 0.6 * np.random.rand()

    return spectrum, class_idx

# Generate dataset
x = np.linspace(0.3, 2.5, num_features)
X_realistic = np.zeros((num_samples, num_features))
y_realistic = np.zeros(num_samples, dtype=int)

# Uneven class distribution (more realistic)
class_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
for i in range(num_samples):
    class_idx = np.random.choice(num_classes, p=class_weights)
    spectrum, label = generate_spectrum_and_label(class_idx, x)
    X_realistic[i] = spectrum
    y_realistic[i] = label

# Save dataset
np.save("exoplanet_spectra.npy", X_realistic)
np.save("exoplanet_labels.npy", y_realistic)

# Plot examples
plt.figure(figsize=(15, 10))
for class_idx in range(num_classes):
    plt.subplot(2, 3, class_idx + 1)
    samples = np.where(y_realistic == class_idx)[0][:3]
    for sample in samples:
        plt.plot(x, X_realistic[sample], alpha=0.7, 
                label=f"{planet_types[class_idx]}" if sample == samples[0] else "")
    plt.title(f"{planet_types[class_idx]} Spectra")
    plt.xlabel("Wavelength (micron)")
    plt.ylabel("Normalized Intensity")
    plt.legend()

plt.tight_layout()
plt.show()

print(f"Dataset created with {num_samples} samples")
print("\nClass distribution:")
for i, ptype in enumerate(planet_types):
    count = np.sum(y_realistic == i)
    print(f"- {ptype}: {count} samples ({count/num_samples*100:.1f}%)")
