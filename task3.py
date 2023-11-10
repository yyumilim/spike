import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

file_path = 'indy_20161005_06.mat'

with h5py.File(file_path, 'r') as file:
    chan_names = file['chan_names'][:]
    cursor_pos = file['cursor_pos'][:]
    finger_pos = file['finger_pos'][:]
    spikes = file['spikes'][:]
    t = file['t'][:]
    target_pos = file['target_pos'][:]
    wf = file['wf'][:]


# 1. 卡尔曼滤波器 - 只使用位置

A = np.eye(2)
H = np.eye(2)
Q = np.eye(2) * 0.001
R = np.eye(2) * 0.001
x_estimate = cursor_pos[:, 0]
P_estimate = np.eye(2) * 1
estimated_positions = np.zeros_like(cursor_pos)

for i in range(cursor_pos.shape[1]):
    # 预测
    x_predict = A @ x_estimate
    P_predict = A @ P_estimate @ A.T + Q
    # 验证
    K = P_predict @ H.T @ np.linalg.inv(H @ P_predict @ H.T + R)
    x_estimate = x_predict + K @ (cursor_pos[:, i] - H @ x_predict)
    P_estimate = (np.eye(2) - K @ H) @ P_predict

    estimated_positions[:, i] = x_estimate


plt.figure(figsize=(14, 7))
plt.plot(cursor_pos[0, :1000], cursor_pos[1, :1000], label='Actual Position', color='blue', linewidth=1)
plt.plot(estimated_positions[0, :1000], estimated_positions[1, :1000], label='Estimated Position', color='red', linestyle='--', linewidth=1)
plt.title('Cursor Position: Actual vs Estimated')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.legend()
plt.grid(True)
plt.show()                  #前1000个


plt.figure(figsize=(14, 7))
plt.plot(cursor_pos[0, :], cursor_pos[1, :], label='Actual Position', color='blue', linewidth=1)
plt.plot(estimated_positions[0, :], estimated_positions[1, :], label='Estimated Position', color='red', linestyle='--', linewidth=1)
plt.title('Cursor Position: Actual vs Estimated (Entire Dataset)')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.legend()
plt.grid(True)
plt.show()  # all

########################################################
# 2. 设置不同QRP
def kalman_filter(Q_val, R_val, P_val):
    Q = np.eye(2) * Q_val
    R = np.eye(2) * R_val
    P_estimate = np.eye(2) * P_val
    x_estimate = cursor_pos[:, 0]
    estimated_positions = np.zeros_like(cursor_pos)

    for i in range(cursor_pos.shape[1]):
        x_predict = A @ x_estimate
        P_predict = A @ P_estimate @ A.T + Q
        K = P_predict @ H.T @ np.linalg.inv(H @ P_predict @ H.T + R)
        x_estimate = x_predict + K @ (cursor_pos[:, i] - H @ x_predict)
        P_estimate = (np.eye(2) - K @ H) @ P_predict
        estimated_positions[:, i] = x_estimate
    return estimated_positions


# 五个不同 Q, R, P
parameters = [
    (0.001, 0.001, 1),
    (0.1, 0.1, 10),
    (1000, 1000, 0.001),
    (1, 1, 10),
    (0.001, 0.001, 1000),
]

estimations = {}
for i, (Q_val, R_val, P_val) in enumerate(parameters):
    estimations[f'Q={Q_val}, R={R_val}, P={P_val}'] = kalman_filter(Q_val, R_val, P_val)

data_subset_length = 5000

def kalman_filter_subset(Q_val, R_val, P_val, data_length):
    Q = np.eye(2) * Q_val
    R = np.eye(2) * R_val
    P_estimate = np.eye(2) * P_val
    x_estimate = cursor_pos[:, 0]
    estimated_positions = np.zeros((2, data_length))

    for i in range(data_length):
        x_predict = A @ x_estimate
        P_predict = A @ P_estimate @ A.T + Q
        K = P_predict @ H.T @ np.linalg.inv(H @ P_predict @ H.T + R)
        x_estimate = x_predict + K @ (cursor_pos[:, i] - H @ x_predict)
        P_estimate = (np.eye(2) - K @ H) @ P_predict
        estimated_positions[:, i] = x_estimate
    return estimated_positions

estimations_subset = {}
for i, (Q_val, R_val, P_val) in enumerate(parameters):
    estimations_subset[f'Q={Q_val}, R={R_val}, P={P_val}'] = kalman_filter_subset(Q_val, R_val, P_val, data_subset_length)

colors = ['blue', 'green', 'magenta', 'cyan', 'orange']
plt.figure(figsize=(14, 14))

for i, (desc, estimated_positions_subset) in enumerate(estimations_subset.items()):
    plt.subplot(len(parameters), 1, i+1)
    plt.plot(cursor_pos[0, :data_subset_length], cursor_pos[1, :data_subset_length], label='Actual Position', color='black', linewidth=1)
    plt.plot(estimated_positions_subset[0, :], estimated_positions_subset[1, :], label=f'Estimated Position ({desc})', linestyle='--', color=colors[i], linewidth=1)
    plt.title(f'Parameter Set: {desc}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

#################################################
# 只使用位置、速度
delta_t = np.diff(t.flatten())
mean_delta_t = np.mean(delta_t)
A_extended = np.array([[1, 0, mean_delta_t, 0],
                       [0, 1, 0, mean_delta_t],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

H_extended = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]])

Q_extended = np.eye(4) * 0.001
R_extended = np.eye(2) * 0.001
x_estimate_extended = np.array([cursor_pos[0, 0], cursor_pos[1, 0], 0, 0])
P_estimate_extended = np.eye(4)
P_estimate_extended[2:, 2:] *= 10
estimated_positions_extended = np.zeros((2, data_subset_length))
estimated_velocities_extended = np.zeros((2, data_subset_length))

for i in range(data_subset_length):
    x_predict_extended = A_extended @ x_estimate_extended
    P_predict_extended = A_extended @ P_estimate_extended @ A_extended.T + Q_extended
    K_extended = P_predict_extended @ H_extended.T @ np.linalg.inv(H_extended @ P_predict_extended @ H_extended.T + R_extended)
    x_estimate_extended = x_predict_extended + K_extended @ (cursor_pos[:, i] - H_extended @ x_predict_extended)
    P_estimate_extended = (np.eye(4) - K_extended @ H_extended) @ P_predict_extended

    estimated_positions_extended[:, i] = x_estimate_extended[:2]
    estimated_velocities_extended[:, i] = x_estimate_extended[2:]

plt.figure(figsize=(14, 7))
plt.plot(cursor_pos[0, :data_subset_length], cursor_pos[1, :data_subset_length], label='Actual Position', color='black', linewidth=1)
plt.plot(estimated_positions_extended[0, :], estimated_positions_extended[1, :], label='Estimated Position (with velocity)', linestyle='--', color='orange', linewidth=1)

plt.title('Extended Kalman Filter with Position and Velocity')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



# MSE
mse_position_only = mean_squared_error(cursor_pos[:, :data_subset_length].T, estimated_positions[:, :data_subset_length].T)
print("MSE for the position-only Kalman filter: ",mse_position_only)
mse_position_velocity = mean_squared_error(cursor_pos[:, :data_subset_length].T, estimated_positions_extended.T)

mse_values = [mse_position_only, mse_position_velocity]
labels = ['Position Only', 'Position + Velocity']

plt.figure(figsize=(10, 6))
plt.bar(labels, mse_values, color=['blue', 'orange'])
plt.title('Comparison of MSE - Position Only vs Position + Velocity')
plt.ylabel('Mean Squared Error')
plt.yscale('log')  # Log scale to better visualize the large difference in errors
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()


####################### 加速度
A_extended_acc = np.array([
    [1, 0, mean_delta_t, 0, 0.5 * mean_delta_t ** 2, 0],
    [0, 1, 0, mean_delta_t, 0, 0.5 * mean_delta_t ** 2],
    [0, 0, 1, 0, mean_delta_t, 0],
    [0, 0, 0, 1, 0, mean_delta_t],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])


H_extended_acc = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]])
Q_extended_acc = np.eye(6) * 0.001
Q_extended_acc[4:, 4:] *= 10
R_extended_acc = np.eye(2) * 0.001
x_estimate_extended_acc = np.array([cursor_pos[0, 0], cursor_pos[1, 0], 0, 0, 0, 0])
P_estimate_extended_acc = np.eye(6)
P_estimate_extended_acc[2:4, 2:4] *= 10
P_estimate_extended_acc[4:, 4:] *= 100

estimated_positions_extended_acc = np.zeros((2, data_subset_length))

for i in range(data_subset_length):
    x_predict_extended_acc = A_extended_acc @ x_estimate_extended_acc
    P_predict_extended_acc = A_extended_acc @ P_estimate_extended_acc @ A_extended_acc.T + Q_extended_acc

    K_extended_acc = P_predict_extended_acc @ H_extended_acc.T @ np.linalg.inv( H_extended_acc @ P_predict_extended_acc @ H_extended_acc.T + R_extended_acc)
    x_estimate_extended_acc = x_predict_extended_acc + K_extended_acc @ ( cursor_pos[:, i] - H_extended_acc @ x_predict_extended_acc)
    P_estimate_extended_acc = (np.eye(6) - K_extended_acc @ H_extended_acc) @ P_predict_extended_acc

    estimated_positions_extended_acc[:, i] = x_estimate_extended_acc[:2]

mse_position_velocity_acc = mean_squared_error(cursor_pos[:, :data_subset_length].T, estimated_positions_extended_acc.T)
plt.figure(figsize=(14, 7))

plt.plot(cursor_pos[0, :data_subset_length], cursor_pos[1, :data_subset_length], label='Actual Position', color='black', linewidth=1)
plt.plot(estimated_positions_extended[0, :], estimated_positions_extended[1, :], label='Estimated Position (with velocity)', linestyle='--', color='orange', linewidth=1)
plt.plot(estimated_positions_extended_acc[0, :], estimated_positions_extended_acc[1, :], label='Estimated Position (with velocity and acceleration)', linestyle='--', color='green', linewidth=1)

plt.title('Extended Kalman Filter with Position, Velocity, and Acceleration')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# MSE
mse_values = [mse_position_only, mse_position_velocity, mse_position_velocity_acc]
labels = ['Position Only', 'Position + Velocity', 'Position + Velocity + Acceleration']

plt.figure(figsize=(10, 6))
plt.bar(labels, mse_values, color=['blue', 'orange', 'green'])
plt.title('Comparison of MSE for Different Kalman Filter Models')
plt.ylabel('Mean Squared Error')
plt.yscale('log')  # Log scale to better visualize the differences in errors
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()


########################################## 现性回归

def create_features_labels(positions, num_past_points=5):
    features = []
    labels = []
    for i in range(num_past_points, len(positions[0])):
        feature = positions[:, i - num_past_points:i].flatten()
        label = positions[:, i]
        features.append(feature)
        labels.append(label)

    return np.array(features), np.array(labels)


features, labels = create_features_labels(cursor_pos[:, :data_subset_length], num_past_points=5)
split_index = int(features.shape[0] * 0.8)
features_train, features_test = features[:split_index], features[split_index:]
labels_train, labels_test = labels[:split_index], labels[split_index:]

linear_model = LinearRegression()
linear_model.fit(features_train, labels_train)
predictions_linear = linear_model.predict(features_test)
mse_linear_regression = mean_squared_error(labels_test, predictions_linear)
print("MSE for the position-only Linear Regression: ",mse_linear_regression)



mse_linear_kalman = [mse_position_only, mse_linear_regression]
labels = ['Kalman filter', 'Linear Regression']

plt.figure(figsize=(10, 6))
plt.bar(labels, mse_linear_kalman, color=['blue', 'orange'])
plt.title('Comparison of MSE - Kalman filter vs Linear Regression')
plt.ylabel('Mean Squared Error')
plt.yscale('log')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()