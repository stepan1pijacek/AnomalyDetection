import numpy as np


class DataAugmentaion :
    def __init__(self, noise_level, drift_factor, 
            scale_factor, num_augmented_samples) -> None:
        self.num_augmented_samples = num_augmented_samples
        self.noise_level = noise_level
        self.drift_factor = drift_factor
        self.scale_factor = scale_factor

    def apply_data_augmentaion (self, original_data) :
        augmented_data = []

        for spectrum in original_data:
            for _ in range(self.num_augmented_samples):
                augmented_spectrum = spectrum.copy()

                augmented_spectrum = self.add_noise(augmented_spectrum)
                augmented_spectrum = self.apply_baseline_drift(augmented_spectrum)
                augmented_spectrum = self.scale_intensity(augmented_spectrum)

                augmented_data.append(augmented_spectrum)

        return np.array(augmented_data)

    def add_noise(self, spectrum) :
        noise = np.random.normal(0, self.noise_level, spectrum.shape)
        return spectrum + noise

    def apply_baseline_drift(self, spectrum) :
        baseline_drift = np.linspace(0, self.drift_factor * len(spectrum), len(spectrum))
        return spectrum + baseline_drift

    def scale_intensity(self, spectrum) :
        return spectrum * self.scale_factor