import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.linalg import block_diag

file_path = 'indy_20161005_06.mat'



# 1. 绘制raster图
def plot_raster_batch_adjusted(all_units, channel_range, batch_number, start_time, title='Raster Plot Batch'):
    num_plots = sum(len(all_units[ch]) for ch in channel_range)
    fig, axs = plt.subplots(num_plots, 1, figsize=(20, 0.5 * num_plots), sharex=True)

    unit_count = 0
    for channel_index in channel_range:
        channel_units = all_units[channel_index]
        for unit_index, spikes in enumerate(channel_units):
            spikes = spikes[spikes >= start_time]
            y = [unit_count] * len(spikes)
            axs[unit_count].scatter(spikes, y, marker='|', color='black')
            axs[unit_count].set_yticks([])
            axs[unit_count].grid(False)
            for spine in axs[unit_count].spines.values():
                spine.set_visible(False)
            unit_count += 1

    fig.suptitle(f'All channel', fontsize=16)
    axs[0].set_xticks([])
    plt.subplots_adjust(hspace=0)
    plt.show()

def extract_all_sorted_units(file, spikes_data):
    all_sorted_unit_spike_times = []
    for channel_index in range(spikes_data.shape[1]):
        channel_sorted_units = []
        for unit_index in range(1, spikes_data.shape[0]):
            ref = spikes_data[unit_index, channel_index]
            if ref:
                spikes = file[ref][:].flatten()
                if spikes.size > 0:
                    channel_sorted_units.append(spikes)
        all_sorted_unit_spike_times.append(channel_sorted_units)
    return all_sorted_unit_spike_times


with h5py.File(file_path, 'r') as file:
    chan_names = file['chan_names'][:]
    cursor_pos = file['cursor_pos'][:]
    finger_pos = file['finger_pos'][:]
    spikes = file['spikes'][:]
    t = file['t'][:]
    target_pos = file['target_pos'][:]
    wf = file['wf'][:]

    all_sorted_spike_times = extract_all_sorted_units(file, file['spikes'])
    all_units_spikes = []

    for i in range(spikes.shape[0]):
        unit_spikes = []
        for j in range(spikes.shape[1]):
            ref = spikes[i, j]
            if ref:
                spikes_data = file[ref][:]
                unit_spikes.append(spikes_data)
        all_units_spikes.append(unit_spikes)


num_channels = len(all_sorted_spike_times)
num_sorted_units = sum(len(channel_units) for channel_units in all_sorted_spike_times)
batch_size = 96
num_batches = (num_channels // batch_size) + (0 if num_channels % batch_size == 0 else 1)  # Total number of batches needed
batch_number = 1
channel_range = range((batch_number - 1) * batch_size, min(batch_number * batch_size, num_channels))
start_time = 1250
plot_raster_batch_adjusted(all_sorted_spike_times, channel_range, batch_number, start_time)



#############################################################################################################

# 2. 绘制tuning curve图 (只有位置)
target_x_positions = target_pos[0, :]
target_y_positions = target_pos[1, :]
angles = np.degrees(np.arctan2(target_y_positions, target_x_positions))
angles = np.mod(angles, 360)
num_direction_bins = 8
bin_size = 360 / num_direction_bins
direction_bins = np.arange(0, 360, bin_size)
hist, edges = np.histogram(angles, bins=direction_bins)
bin_centers = (edges[:-1] + edges[1:]) / 2

# 使用UnivariateSpline进行平滑拟合
spline = UnivariateSpline(bin_centers, hist, s=0.5)  # s是平滑因子
x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 300)
y_smooth = spline(x_smooth)

plt.plot(x_smooth, y_smooth, label='Tuning Curve')
plt.title('Tuning Curve of Movement Directions')
plt.xlabel('Direction (degrees)')
plt.ylabel('Frequency')
plt.xticks(direction_bins)
plt.legend()
plt.show()


############################################################ tuning curve agn

# 3. 绘制tuning curve图 (速度)

t = np.ravel(t)
delta_pos = np.diff(cursor_pos, axis=1)
delta_t = np.diff(t)
delta_t[delta_t == 0] = np.finfo(float).eps
velocity = np.sqrt((delta_pos[0] / delta_t) ** 2 + (delta_pos[1] / delta_t) ** 2)

finger_pos_mm = finger_pos * 10
finger_pos_xy = finger_pos_mm[:2, :]

# 计算速度，即位置的时间导数
delta_finger_pos = np.diff(finger_pos_xy, axis=1)
delta_t[delta_t == 0] = np.finfo(float).eps
finger_velocity = np.sqrt((delta_finger_pos[0] / delta_t) ** 2 + (delta_finger_pos[1] / delta_t) ** 2)

# 绘制手指速度的直方图
plt.hist(finger_velocity, bins=50, alpha=0.75)
plt.title('Finger Velocity Distribution')
plt.xlabel('Velocity (mm/s)')
plt.ylabel('Frequency')
plt.show()


sorted_units_spikes = all_units_spikes[1:]
total_time = t[-1] - t[0]
discharge_rates = []
for unit in sorted_units_spikes:
    unit_rates = []
    for channel_spikes in unit:
        rate = len(channel_spikes) / total_time
        unit_rates.append(rate)
    discharge_rates.append(unit_rates)


num_bins = 20
velocity_bins, bin_edges = np.histogram(finger_velocity, bins=num_bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

sorted_spike_times_flat = np.concatenate([
    spikes.flatten() for unit in sorted_units_spikes[1:] for channel in unit for spikes in channel if spikes.size > 0
])
sorted_spike_indices = np.searchsorted(t[1:], sorted_spike_times_flat)
sorted_spike_indices = sorted_spike_indices[sorted_spike_indices < len(finger_velocity)]

sorted_spike_bins = np.digitize(finger_velocity[sorted_spike_indices], bin_edges) - 1
sorted_spikes_per_bin = np.zeros(num_bins)
sorted_time_per_bin = np.zeros(num_bins)

for idx, bin_idx in enumerate(sorted_spike_bins):
    if 0 <= bin_idx < num_bins:
        sorted_spikes_per_bin[bin_idx] += 1
        sorted_time_per_bin[bin_idx] += delta_t[sorted_spike_indices[idx] - 1]

sorted_time_per_bin[sorted_time_per_bin == 0] = np.finfo(float).eps
sorted_average_rates_per_bin = sorted_spikes_per_bin / sorted_time_per_bin

plt.plot(bin_centers, sorted_average_rates_per_bin, marker='o')
plt.title('Neuronal Tuning Curve Based on Finger Velocity (Sorted Units Only)')
plt.xlabel('Finger Velocity (mm/s)')
plt.ylabel('Average Discharge Rate (spikes/s)')
plt.show()

##################################################################
# 4. 对速度编码模型的拟合程度

y = sorted_average_rates_per_bin  # 放电率
X = bin_centers.reshape(-1, 1)  # 速度值

# 建立线性回归模型
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)   # 计算预测值
r_squared = r2_score(y, y_pred)     # 计算R^2
print("r_squared = ", r_squared)    # 输出模型的R^2

# 绘制实际放电率和模型预测放电率的关系
plt.scatter(X, y, color='black', label='Actual Discharge Rates')
plt.plot(X, y_pred, color='blue', linewidth=2, label='Predicted Discharge Rates')
plt.title('Neuronal Discharge Rates vs. Velocity')
plt.xlabel('Velocity (mm/s)')
plt.ylabel('Discharge Rate (spikes/s)')
plt.legend()
plt.show()

##################################################################################

# 5. 绘制PSTH直方图
pre_stimulus_time = 1.0
post_stimulus_time = 1.0
bin_width = 0.01  #
bins = np.arange(-pre_stimulus_time, post_stimulus_time, bin_width)
bin_centers = (bins[:-1] + bins[1:]) / 2

adjusted_spike_times = sorted_spike_times_flat - t[0]
psth_counts, _ = np.histogram(adjusted_spike_times, bins=bins)
psth_rates = psth_counts / (bin_width * len(sorted_units_spikes[1:]))  # 归一化频率

plt.bar(bin_centers, psth_rates, width=bin_width, color='grey')
plt.title('PSTH of Neuronal Discharge Rates')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()

########################################################################################
# 6. 绘制tuning curve图 (加速度)
# 加速度是速度的时间导数，可以通过对速度进行差分来估算
acceleration = np.diff(finger_velocity) / np.mean(delta_t)

adjusted_spike_indices = sorted_spike_indices[sorted_spike_indices < len(acceleration) + 1]

acceleration_bins = np.linspace(np.min(acceleration), np.max(acceleration), num_bins + 1)
acceleration_bin_centers = (acceleration_bins[:-1] + acceleration_bins[1:]) / 2

spikes_per_acceleration_bin = np.zeros(num_bins)
time_in_acceleration_bins = np.zeros(num_bins)

for spike_index in adjusted_spike_indices:
    acceleration_value = acceleration[spike_index - 1]  #
    bin_index = np.digitize(acceleration_value, acceleration_bins) - 1
    if 0 <= bin_index < num_bins:
        spikes_per_acceleration_bin[bin_index] += 1
        time_in_acceleration_bins[bin_index] += delta_t[spike_index - 1]

time_in_acceleration_bins[time_in_acceleration_bins == 0] = np.finfo(float).eps
average_rates_per_acceleration_bin = spikes_per_acceleration_bin / time_in_acceleration_bins

plt.plot(acceleration_bin_centers, average_rates_per_acceleration_bin, marker='o')
plt.title('Neuronal Tuning Curve Based on Acceleration')
plt.xlabel('Acceleration (mm/s²)')
plt.ylabel('Average Discharge Rate (spikes/s)')
plt.show()
