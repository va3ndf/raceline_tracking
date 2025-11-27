import math
import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

# Global configuration for the controller
CONTROLLER_CONFIG = {
    "steering_kp": 8.0315,
    "steering_ki": 0.0000,
    "steering_kd": 0.9600,
    "velocity_kp": 127.3380,
    "velocity_ki": 0.0827,
    "velocity_kd": 3.6845,
    "lookahead_k": 0.1264,
    "lookahead_L0": 9.5894,
    "braking_factor": 1.1098,
    "steer_limit_factor": 0.5000,
    "lookahead_brake_scale": 3.1451,
}

def get_next_point(
    state : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    x, y, steering_angle, lng_velocity, heading = state

    # Find the closest point on the racetrack centerline
    centerline = racetrack.centerline
    diffs = centerline - np.array([x, y])
    dists = np.linalg.norm(diffs, axis=1)
    closest_index = np.argmin(dists)

    # Pure Pursuit Lookahead
    # Lookahead distance Ld
    # Ld = k * v + L0
    k = CONTROLLER_CONFIG["lookahead_k"]
    L0 = CONTROLLER_CONFIG["lookahead_L0"]
    Ld = k * lng_velocity + L0
    
    # Find point at distance Ld
    # Simple approximation: accumulate distances along path
    current_dist = 0.0
    next_index = closest_index
    while current_dist < Ld:
        next_index = (next_index + 1) % len(centerline)
        current_dist += np.linalg.norm(centerline[next_index] - centerline[next_index-1])
        if next_index == closest_index: # Looped around
            break
            
    return centerline[next_index]

class LowerController:
    def __init__(self):
        self.prev_steering_error = 0.0
        self.integral_steering_error = 0.0
        self.prev_velocity_error = 0.0
        self.integral_velocity_error = 0.0

        self.steering_kp = CONTROLLER_CONFIG["steering_kp"]
        self.steering_ki = CONTROLLER_CONFIG["steering_ki"]
        self.steering_kd = CONTROLLER_CONFIG["steering_kd"]

        self.velocity_kp = CONTROLLER_CONFIG["velocity_kp"]
        self.velocity_ki = CONTROLLER_CONFIG["velocity_ki"]
        self.velocity_kd = CONTROLLER_CONFIG["velocity_kd"]

    # Functor
    def __call__(self, state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
        # [steer angle, velocity]
        assert(desired.shape == (2,))

        x, y, steering_angle, lng_velocity, heading = state
        desired_angle, desired_velocity = desired

        # Steering PID
        # We want to control the steering angle to match desired_angle
        # The output is steering_velocity (u1)
        
        steering_error = desired_angle - steering_angle
        self.integral_steering_error += steering_error
        steering_derivative = steering_error - self.prev_steering_error
        self.prev_steering_error = steering_error

        steering_control = (self.steering_kp * steering_error + 
                            self.steering_ki * self.integral_steering_error + 
                            self.steering_kd * steering_derivative)

        # Velocity PID
        # We want to control velocity to match desired_velocity
        # The output is acceleration (u2)
        
        velocity_error = desired_velocity - lng_velocity
        self.integral_velocity_error += velocity_error
        velocity_derivative = velocity_error - self.prev_velocity_error
        self.prev_velocity_error = velocity_error

        velocity_control = (self.velocity_kp * velocity_error + 
                            self.velocity_ki * self.integral_velocity_error + 
                            self.velocity_kd * velocity_derivative)

        return np.array([steering_control, velocity_control]).T

lower_controller = LowerController()

def compute_curvature(racetrack: RaceTrack, points=None):
    if points is None:
        points = racetrack.centerline
        attr_name = 'curvature'
    else:
        attr_name = 'path_curvature'

    if hasattr(racetrack, attr_name):
        return

    n_points = len(points)
    curvature = np.zeros(n_points)

    for i in range(n_points):
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[(i + 1) % n_points]

        # Circumcircle radius
        # a, b, c are side lengths
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p2 - p3)
        c = np.linalg.norm(p3 - p1)

        # Area of triangle using Heron's formula
        s = (a + b + c) / 2.0
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))

        if area < 1e-6:
            curvature[i] = 0.0
        else:
            R = (a * b * c) / (4.0 * area)
            curvature[i] = 1.0 / R
            
    setattr(racetrack, attr_name, curvature)

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    x, y, steering_angle, lng_velocity, heading = state

    # Select path
    if racetrack.raceline is not None:
        path_points = racetrack.raceline
        compute_curvature(racetrack, path_points)
        curvature = racetrack.path_curvature
    else:
        path_points = racetrack.centerline
        compute_curvature(racetrack)
        curvature = racetrack.curvature

    # Find the closest point on the path
    diffs = path_points - np.array([x, y])
    dists = np.linalg.norm(diffs, axis=1)
    closest_index = np.argmin(dists)

    # Pure Pursuit Lookahead
    # Lookahead distance Ld
    # Ld = k * v + L0
    k = CONTROLLER_CONFIG["lookahead_k"]
    L0 = CONTROLLER_CONFIG["lookahead_L0"]
    Ld = k * lng_velocity + L0
    
    # Find point at distance Ld
    # Simple approximation: accumulate distances along path
    current_dist = 0.0
    next_index = closest_index
    while current_dist < Ld:
        next_index = (next_index + 1) % len(path_points)
        current_dist += np.linalg.norm(path_points[next_index] - path_points[next_index-1])
        if next_index == closest_index: # Looped around
            break
            
    target_point = path_points[next_index]
    
    # Pure Pursuit Steering Control
    # alpha is the angle between the vehicle's heading and the lookahead vector
    lookahead_vector = target_point - np.array([x, y])
    lookahead_angle = np.arctan2(lookahead_vector[1], lookahead_vector[0])
    
    alpha = lookahead_angle - heading
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi # Normalize
    
    # Ld is the distance to the lookahead point
    Ld = np.linalg.norm(lookahead_vector)
    
    # Steering angle delta = arctan(2L sin(alpha) / Ld)
    # L is wheelbase (parameters[0])
    wheelbase = parameters[0]
    desired_angle = np.arctan2(2 * wheelbase * np.sin(alpha), Ld)
    
    # Clamp desired angle
    max_steer = parameters[4] # max_steering_angle
    desired_angle = np.clip(desired_angle, -max_steer, max_steer)

    # Velocity Control: Curvature-based
    
    # Extract parameters
    max_velocity = parameters[5]
    max_accel = parameters[10]
    max_steer_vel = parameters[9]
    wheelbase = parameters[0]
    
    current_steering_angle = state[2]
    
    # Look ahead for max curvature
    lookahead_dist = (max_velocity**2) / (2 * max_accel) * CONTROLLER_CONFIG["lookahead_brake_scale"]
    
    current_dist = 0.0
    idx = closest_index
    
    min_v_target = max_velocity
    
    # Safety margin for braking
    effective_braking_accel = max_accel * CONTROLLER_CONFIG["braking_factor"]
    
    # Track steering angle along the path
    # Start from the path's required steering, not current car steering
    # This prevents slowing down just because we are correcting an error
    prev_steering_req = np.arctan(wheelbase * curvature[idx])
    
    while current_dist < lookahead_dist:
        prev_idx = idx
        idx = (idx + 1) % len(path_points)
        
        dist_step = np.linalg.norm(path_points[idx] - path_points[prev_idx])
        if dist_step < 1e-3: continue
        
        k = curvature[idx]
        
        # Required steering angle for this curvature
        steering_req = np.arctan(wheelbase * k)
        
        # Limit based on changing steering from previous point to this point
        delta_diff = abs(steering_req - prev_steering_req)
        
        # Ignore small steering changes (noise)
        if delta_diff > 1e-3:
            v_steer_limit = (dist_step * max_steer_vel * CONTROLLER_CONFIG["steer_limit_factor"]) / delta_diff
        else:
            v_steer_limit = max_velocity
            
        # Update prev for next step
        prev_steering_req = steering_req
        
        v_corner = min(max_velocity, v_steer_limit)
        
        # Calculate max velocity we can have NOW
        v_allowable = np.sqrt(v_corner**2 + 2 * effective_braking_accel * current_dist)
        
        if v_allowable < min_v_target:
            min_v_target = v_allowable
        
        current_dist += dist_step
        if idx == closest_index: break
            
    desired_velocity = min(max_velocity, min_v_target)
    desired_velocity = max(desired_velocity, 0.0)
    
    return np.array([desired_angle, desired_velocity]).T
