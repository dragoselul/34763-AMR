import numpy as np
from scipy.linalg import block_diag

class EKFTracker:
    def __init__(self, x0, P0, motion_model, frame_manager, initial_time=0.0):
        self.x = x0
        self.P = P0
        self.motion_model = motion_model
        self.frame_manager = frame_manager
        self.state_history = [self.x.copy()]
        self.time_history = [initial_time]

    def predict(self):
        F = self.motion_model.F
        Q = self.motion_model.Q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update_joint(self, sensor_id_list, measurements, time=None):
        
        z_list = []
        z_pred_list = []
        H_list = []
        R_list = []

        for sensor_id, z in zip(sensor_id_list, measurements):
            if sensor_id == "AIS":
                return self.update_ais(z, time)
            
            z_pred = self.frame_manager.compute_measurement(sensor_id, self.x)
            H = self.frame_manager.compute_jacobian(sensor_id, self.x)
            R = self.frame_manager.get_noise_covariance(sensor_id)
            z_list.append(z)
            z_pred_list.append(z_pred)
            H_list.append(H)
            R_list.append(R)

        z = np.concatenate(z_list)
        z_pred = np.concatenate(z_pred_list)
        H = np.vstack(H_list)
        R = block_diag(*R_list)

        y = z - z_pred
        for i in range(1, len(y), 2):
            y[i] = np.arctan2(np.sin(y[i]), np.cos(y[i]))  

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.state_history.append(self.x.copy())
        self.time_history.append(time)


    def update(self, sensor_id, z, time=None):
        if sensor_id == "AIS":
            return self.update_ais(z, time)
        
        z_pred = self.frame_manager.compute_measurement(sensor_id, self.x)
        H = self.frame_manager.compute_jacobian(sensor_id, self.x)
        R = self.frame_manager.get_noise_covariance(sensor_id)

        if z_pred is None or H is None or R is None:
            return None

        y = z - z_pred
        y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))  

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        nis = float(y.T @ np.linalg.inv(S) @ y)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.state_history.append(self.x.copy())
        self.time_history.append(time)
        return nis

    def update_ais(self, z, time=None):
        z = np.asarray(z, dtype=float)

        z_pred = self.x[:2]

        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        R = self.frame_manager.get_noise_covariance("AIS")

        y = z - z_pred

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        nis = float(y.T @ np.linalg.inv(S) @ y)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

        self.state_history.append(self.x.copy())
        self.time_history.append(time)

        return nis

    def get_rmse(self, ground_truth):
        estimates = np.asarray(self.state_history, dtype=float)
        truth = np.asarray(ground_truth, dtype=float)

        if truth.ndim != 2 or truth.shape[1] < 3:
            raise ValueError("ground_truth must contain rows with at least [time, p_north, p_east].")

        if len(self.time_history) == len(estimates) and all(time is not None for time in self.time_history):
            true_positions = []
            for estimate_time in self.time_history:
                truth_index = int(np.argmin(np.abs(truth[:, 0] - estimate_time)))
                true_positions.append(truth[truth_index, 1:3])

            true_positions = np.asarray(true_positions, dtype=float)
            estimated_positions = estimates[:, :2]
            squared_position_errors = np.sum((estimated_positions - true_positions) ** 2, axis=1)

            return float(np.sqrt(np.mean(squared_position_errors)))

        count = min(len(estimates), len(truth))
        if count == 0:
            raise ValueError("Cannot compute RMSE without estimates and ground truth.")

        estimated_positions = estimates[:count, :2]
        true_positions = truth[:count, 1:3]
        squared_position_errors = np.sum((estimated_positions - true_positions) ** 2, axis=1)

        return float(np.sqrt(np.mean(squared_position_errors)))
    
    
