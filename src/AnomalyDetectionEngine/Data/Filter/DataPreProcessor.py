import matplotlib.pyplot as plt
import numpy as np

from skimage import restoration
from scipy.signal import savgol_filter

class Filter :
    def clean_data(self, data_array_x, data_array_y, ball_radius, passes,
                        plotting, normalization_peak, window_length, polyorder) :

        data_array_y_flipped = np.flip(data_array_y)
        data_array_x_flipped = np.flip(data_array_x)

        temp_data_array = ([])
        normalized_data_array = data_array_y_flipped

        for _ in range(passes):
            temp_data_array = restoration.rolling_ball(data_array_y_flipped, radius=ball_radius)

        filtered_data = np.subtract(normalized_data_array, temp_data_array)
        filtered_data = savgol_filter(filtered_data, window_length, polyorder)

        filtered_data = self.normalization(filtered_data, normalization_peak)

        if plotting:
            plt.figure()

            plt.plot(data_array_x_flipped, normalized_data_array, label='original')
            plt.plot(data_array_x_flipped, filtered_data, label='radius=1000, passes=10')
            plt.legend()

            plt.show()
            return filtered_data

        return filtered_data

    def normalization(self, data_array, peak):
        frequency_value = np.linspace(min(data_array), max(data_array), len(data_array))

        peak_index = np.argmin(np.abs(frequency_value - peak))

        peak_value = data_array[peak_index]

        normalized_data_array = data_array / peak_value

        return normalized_data_array
