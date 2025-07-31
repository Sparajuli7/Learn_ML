# Robotics Machine Learning

## 🤖 Overview
Machine Learning has revolutionized robotics, enabling autonomous navigation, computer vision, motion planning, and intelligent decision-making. This comprehensive guide covers key applications and implementations in robotics.

---

## 👁️ Computer Vision for Robotics

### Visual Perception Systems
Computer vision enables robots to understand their environment through cameras and sensors.

#### Object Detection and Recognition

```python
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class RobotVisionSystem:
    def __init__(self):
        self.object_detector = self.build_object_detector()
        self.pose_estimator = self.build_pose_estimator()
        self.scene_analyzer = self.build_scene_analyzer()
        
    def build_object_detector(self):
        """Build object detection model for robotics"""
        
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Add custom detection head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(80, activation='softmax')(x)  # COCO classes
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def detect_objects(self, image):
        """Detect objects in robot's field of view"""
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Run detection
        predictions = self.object_detector.predict(processed_image)
        
        # Post-process results
        detected_objects = self.postprocess_detections(predictions)
        
        return detected_objects
    
    def preprocess_image(self, image):
        """Preprocess image for ML model"""
        
        # Resize to model input size
        resized = cv2.resize(image, (224, 224))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def postprocess_detections(self, predictions):
        """Convert model predictions to object detections"""
        
        detected_objects = []
        
        for i, confidence in enumerate(predictions[0]):
            if confidence > 0.5:  # Confidence threshold
                detected_objects.append({
                    'class_id': i,
                    'class_name': self.get_class_name(i),
                    'confidence': confidence,
                    'bounding_box': self.estimate_bounding_box(i, confidence)
                })
        
        return detected_objects
    
    def get_class_name(self, class_id):
        """Get class name from COCO dataset"""
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            # ... more classes
        ]
        return coco_classes[class_id] if class_id < len(coco_classes) else f'class_{class_id}'
    
    def estimate_bounding_box(self, class_id, confidence):
        """Estimate bounding box for detected object"""
        # Simplified bounding box estimation
        # In practice, use models like YOLO or Faster R-CNN
        return {
            'x_min': 0.1,
            'y_min': 0.1,
            'x_max': 0.9,
            'y_max': 0.9,
            'confidence': confidence
        }
```

---

## 🎯 Motion Planning and Control

### Intelligent Path Planning
ML enables robots to plan optimal paths while avoiding obstacles and optimizing for efficiency.

#### Reinforcement Learning for Motion Planning

```python
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim

class RobotEnvironment(gym.Env):
    """Custom environment for robot motion planning"""
    
    def __init__(self, map_size=(10, 10)):
        super(RobotEnvironment, self).__init__()
        
        self.map_size = map_size
        self.robot_pos = [0, 0]
        self.target_pos = [map_size[0]-1, map_size[1]-1]
        self.obstacles = self.generate_obstacles()
        
        # Action space: 4 directions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Observation space: robot position, target position, obstacle map
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(map_size[0] * map_size[1] + 4,)
        )
    
    def generate_obstacles(self):
        """Generate random obstacles in the environment"""
        obstacles = []
        for _ in range(5):  # 5 random obstacles
            x = np.random.randint(0, self.map_size[0])
            y = np.random.randint(0, self.map_size[1])
            if [x, y] != self.robot_pos and [x, y] != self.target_pos:
                obstacles.append([x, y])
        return obstacles
    
    def reset(self):
        """Reset environment to initial state"""
        self.robot_pos = [0, 0]
        return self.get_observation()
    
    def step(self, action):
        """Execute action and return new state"""
        
        # Define action mappings
        action_map = {
            0: [-1, 0],   # Up
            1: [1, 0],    # Down
            2: [0, -1],   # Left
            3: [0, 1]     # Right
        }
        
        # Calculate new position
        new_pos = [
            self.robot_pos[0] + action_map[action][0],
            self.robot_pos[1] + action_map[action][1]
        ]
        
        # Check boundaries
        if (0 <= new_pos[0] < self.map_size[0] and 
            0 <= new_pos[1] < self.map_size[1] and
            new_pos not in self.obstacles):
            self.robot_pos = new_pos
        
        # Calculate reward
        reward = self.calculate_reward()
        
        # Check if episode is done
        done = (self.robot_pos == self.target_pos)
        
        return self.get_observation(), reward, done, {}
    
    def calculate_reward(self):
        """Calculate reward based on current state"""
        
        # Distance to target
        distance = np.linalg.norm(
            np.array(self.robot_pos) - np.array(self.target_pos)
        )
        
        # Reward for reaching target
        if self.robot_pos == self.target_pos:
            return 100
        
        # Penalty for being far from target
        return -distance
    
    def get_observation(self):
        """Get current observation"""
        
        # Create obstacle map
        obstacle_map = np.zeros(self.map_size)
        for obs in self.obstacles:
            obstacle_map[obs[0], obs[1]] = 1
        
        # Flatten obstacle map
        flat_map = obstacle_map.flatten()
        
        # Add robot and target positions
        observation = np.concatenate([
            flat_map,
            self.robot_pos,
            self.target_pos
        ])
        
        return observation

class RobotPolicy(nn.Module):
    """Neural network policy for robot motion planning"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(RobotPolicy, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

class RobotMotionPlanner:
    """RL-based motion planner for robots"""
    
    def __init__(self, env, learning_rate=0.001):
        self.env = env
        self.policy = RobotPolicy(
            input_size=env.observation_space.shape[0],
            hidden_size=64,
            output_size=env.action_space.n
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_episode(self):
        """Train for one episode"""
        
        state = self.env.reset()
        total_reward = 0
        actions_taken = []
        states_visited = []
        
        for step in range(100):  # Max 100 steps per episode
            # Get action probabilities
            state_tensor = torch.FloatTensor(state)
            action_probs = torch.softmax(self.policy(state_tensor), dim=0)
            
            # Sample action
            action = torch.multinomial(action_probs, 1).item()
            
            # Take action
            next_state, reward, done, _ = self.env.step(action)
            
            # Store experience
            actions_taken.append(action)
            states_visited.append(state)
            total_reward += reward
            
            state = next_state
            
            if done:
                break
        
        return total_reward, states_visited, actions_taken
    
    def optimize_policy(self, states, actions, rewards):
        """Optimize policy using collected experience"""
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        
        # Normalize rewards
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # Get action probabilities
        action_probs = torch.softmax(self.policy(states_tensor), dim=1)
        
        # Calculate loss
        loss = self.criterion(action_probs, actions_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

---

## 🔄 Sensor Fusion and State Estimation

### Multi-Sensor Data Integration
Robots use multiple sensors (cameras, LIDAR, IMU) and ML to understand their environment and estimate their state.

#### Kalman Filter with ML Enhancements

```python
import numpy as np
from scipy.linalg import inv
import torch
import torch.nn as nn

class MLEnhancedKalmanFilter:
    """Kalman filter enhanced with ML for better state estimation"""
    
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # Initialize state estimate
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1000  # Initial uncertainty
        
        # Process and measurement noise
        self.Q = np.eye(state_dim) * 0.1  # Process noise
        self.R = np.eye(measurement_dim) * 1.0  # Measurement noise
        
        # ML enhancement network
        self.ml_corrector = self.build_ml_corrector()
    
    def build_ml_corrector(self):
        """Build ML network to correct Kalman filter predictions"""
        
        model = nn.Sequential(
            nn.Linear(self.state_dim + self.measurement_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.state_dim)
        )
        
        return model
    
    def predict(self, u=None):
        """Predict next state"""
        
        # State transition matrix (simplified for 2D position)
        F = np.array([
            [1, 0, 1, 0],  # x position
            [0, 1, 0, 1],  # y position
            [0, 0, 1, 0],  # x velocity
            [0, 0, 0, 1]   # y velocity
        ])
        
        # Predict state
        if u is not None:
            self.x = F @ self.x + u
        else:
            self.x = F @ self.x
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, measurement):
        """Update state estimate with measurement"""
        
        # Measurement matrix
        H = np.array([
            [1, 0, 0, 0],  # Measure x position
            [0, 1, 0, 0]   # Measure y position
        ])
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ inv(S)
        
        # Update state
        y = measurement - H @ self.x  # Innovation
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P
        
        # ML correction
        self.apply_ml_correction(measurement)
    
    def apply_ml_correction(self, measurement):
        """Apply ML-based correction to state estimate"""
        
        # Prepare input for ML network
        ml_input = np.concatenate([self.x, measurement])
        ml_input_tensor = torch.FloatTensor(ml_input).unsqueeze(0)
        
        # Get ML correction
        with torch.no_grad():
            correction = self.ml_corrector(ml_input_tensor).squeeze().numpy()
        
        # Apply correction
        self.x += correction * 0.1  # Small correction factor
    
    def get_state_estimate(self):
        """Get current state estimate"""
        return {
            'position': self.x[:2],
            'velocity': self.x[2:],
            'uncertainty': np.diag(self.P)[:2]
        }

class SensorFusionSystem:
    """Multi-sensor fusion system for robotics"""
    
    def __init__(self):
        self.kalman_filter = MLEnhancedKalmanFilter(state_dim=4, measurement_dim=2)
        self.sensor_weights = {
            'camera': 0.4,
            'lidar': 0.4,
            'imu': 0.2
        }
    
    def fuse_sensor_data(self, camera_data, lidar_data, imu_data):
        """Fuse data from multiple sensors"""
        
        # Process camera data (object detection)
        camera_measurement = self.process_camera_data(camera_data)
        
        # Process LIDAR data (distance measurements)
        lidar_measurement = self.process_lidar_data(lidar_data)
        
        # Process IMU data (orientation and acceleration)
        imu_measurement = self.process_imu_data(imu_data)
        
        # Weighted fusion
        fused_measurement = (
            self.sensor_weights['camera'] * camera_measurement +
            self.sensor_weights['lidar'] * lidar_measurement +
            self.sensor_weights['imu'] * imu_measurement
        )
        
        # Update state estimate
        self.kalman_filter.predict()
        self.kalman_filter.update(fused_measurement)
        
        return self.kalman_filter.get_state_estimate()
    
    def process_camera_data(self, camera_data):
        """Process camera data for position estimation"""
        # Simplified camera processing
        # In practice, use computer vision techniques
        return np.array([camera_data.get('x', 0), camera_data.get('y', 0)])
    
    def process_lidar_data(self, lidar_data):
        """Process LIDAR data for distance measurements"""
        # Simplified LIDAR processing
        # In practice, use point cloud processing
        return np.array([lidar_data.get('distance_x', 0), lidar_data.get('distance_y', 0)])
    
    def process_imu_data(self, imu_data):
        """Process IMU data for orientation and acceleration"""
        # Simplified IMU processing
        # In practice, use quaternion math and sensor fusion
        return np.array([imu_data.get('accel_x', 0), imu_data.get('accel_y', 0)])
```

---

## 🧭 Autonomous Navigation

### SLAM and Path Planning
Simultaneous Localization and Mapping (SLAM) with ML enables robots to navigate unknown environments.

#### Graph-Based SLAM with ML

```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import networkx as nx

class GraphSLAM:
    """Graph-based SLAM with ML enhancements"""
    
    def __init__(self):
        self.poses = []  # Robot poses
        self.landmarks = []  # Landmark positions
        self.measurements = []  # Sensor measurements
        self.graph = nx.Graph()
        
    def add_pose(self, pose_id, x, y, theta):
        """Add robot pose to graph"""
        self.poses.append({
            'id': pose_id,
            'x': x,
            'y': y,
            'theta': theta
        })
        
        # Add node to graph
        self.graph.add_node(pose_id, type='pose', x=x, y=y, theta=theta)
    
    def add_landmark(self, landmark_id, x, y):
        """Add landmark to graph"""
        self.landmarks.append({
            'id': landmark_id,
            'x': x,
            'y': y
        })
        
        # Add node to graph
        self.graph.add_node(landmark_id, type='landmark', x=x, y=y)
    
    def add_measurement(self, from_id, to_id, measurement, measurement_type):
        """Add measurement between nodes"""
        self.measurements.append({
            'from_id': from_id,
            'to_id': to_id,
            'measurement': measurement,
            'type': measurement_type
        })
        
        # Add edge to graph
        self.graph.add_edge(from_id, to_id, 
                           measurement=measurement, 
                           type=measurement_type)
    
    def optimize_graph(self):
        """Optimize graph using least squares"""
        
        # Build information matrix and vector
        H, b = self.build_system()
        
        # Solve linear system
        dx = spsolve(H, b)
        
        # Update poses and landmarks
        self.update_estimates(dx)
    
    def build_system(self):
        """Build linear system for optimization"""
        
        num_poses = len(self.poses)
        num_landmarks = len(self.landmarks)
        total_vars = 3 * num_poses + 2 * num_landmarks
        
        # Initialize information matrix and vector
        H = csr_matrix((total_vars, total_vars))
        b = np.zeros(total_vars)
        
        # Add measurement constraints
        for measurement in self.measurements:
            self.add_measurement_constraint(H, b, measurement)
        
        return H, b
    
    def add_measurement_constraint(self, H, b, measurement):
        """Add measurement constraint to system"""
        
        # Simplified constraint addition
        # In practice, implement proper Jacobian computation
        
        from_id = measurement['from_id']
        to_id = measurement['to_id']
        
        # Get variable indices
        from_idx = self.get_variable_index(from_id)
        to_idx = self.get_variable_index(to_id)
        
        # Add constraint (simplified)
        if from_idx is not None and to_idx is not None:
            H[from_idx, from_idx] += 1
            H[to_idx, to_idx] += 1
            H[from_idx, to_idx] -= 1
            H[to_idx, from_idx] -= 1
    
    def get_variable_index(self, node_id):
        """Get variable index in optimization vector"""
        
        # Find pose index
        for i, pose in enumerate(self.poses):
            if pose['id'] == node_id:
                return 3 * i
        
        # Find landmark index
        for i, landmark in enumerate(self.landmarks):
            if landmark['id'] == node_id:
                return 3 * len(self.poses) + 2 * i
        
        return None
    
    def update_estimates(self, dx):
        """Update pose and landmark estimates"""
        
        # Update poses
        for i, pose in enumerate(self.poses):
            idx = 3 * i
            pose['x'] += dx[idx]
            pose['y'] += dx[idx + 1]
            pose['theta'] += dx[idx + 2]
        
        # Update landmarks
        for i, landmark in enumerate(self.landmarks):
            idx = 3 * len(self.poses) + 2 * i
            landmark['x'] += dx[idx]
            landmark['y'] += dx[idx + 1]

class AutonomousNavigator:
    """Autonomous navigation system for robots"""
    
    def __init__(self):
        self.slam = GraphSLAM()
        self.path_planner = RobotMotionPlanner(RobotEnvironment())
        self.current_pose = {'x': 0, 'y': 0, 'theta': 0}
        self.target_pose = {'x': 10, 'y': 10, 'theta': 0}
    
    def navigate_to_target(self):
        """Navigate to target using SLAM and path planning"""
        
        # Update SLAM with current sensor data
        self.update_slam()
        
        # Plan path to target
        path = self.plan_path()
        
        # Execute path
        for waypoint in path:
            self.move_to_waypoint(waypoint)
        
        return True
    
    def update_slam(self):
        """Update SLAM with current sensor measurements"""
        
        # Add current pose
        pose_id = len(self.slam.poses)
        self.slam.add_pose(pose_id, 
                          self.current_pose['x'],
                          self.current_pose['y'],
                          self.current_pose['theta'])
        
        # Add landmark measurements (simplified)
        landmarks = self.detect_landmarks()
        for landmark in landmarks:
            landmark_id = len(self.slam.landmarks)
            self.slam.add_landmark(landmark_id, landmark['x'], landmark['y'])
            
            # Add measurement
            self.slam.add_measurement(pose_id, landmark_id, 
                                    landmark['measurement'], 'landmark')
        
        # Optimize graph
        self.slam.optimize_graph()
    
    def detect_landmarks(self):
        """Detect landmarks in environment (simplified)"""
        # In practice, use computer vision or LIDAR processing
        return [
            {'x': 2, 'y': 3, 'measurement': {'distance': 3.6, 'angle': 0.5}},
            {'x': 5, 'y': 1, 'measurement': {'distance': 5.1, 'angle': 0.2}}
        ]
    
    def plan_path(self):
        """Plan path to target"""
        
        # Get current and target positions from SLAM
        current_pos = [self.current_pose['x'], self.current_pose['y']]
        target_pos = [self.target_pose['x'], self.target_pose['y']]
        
        # Simple path planning (in practice, use A* or RRT)
        path = [current_pos, target_pos]
        
        return path
    
    def move_to_waypoint(self, waypoint):
        """Move robot to waypoint"""
        
        # Calculate movement command
        dx = waypoint[0] - self.current_pose['x']
        dy = waypoint[1] - self.current_pose['y']
        
        # Update current pose
        self.current_pose['x'] += dx
        self.current_pose['y'] += dy
        
        # In practice, send commands to robot actuators
        print(f"Moving to: {waypoint}")
```

---

## 🚀 Implementation Best Practices

### Robotics ML System Architecture

```python
class RoboticsMLSystem:
    """Complete robotics ML system"""
    
    def __init__(self):
        self.vision_system = RobotVisionSystem()
        self.motion_planner = RobotMotionPlanner(RobotEnvironment())
        self.sensor_fusion = SensorFusionSystem()
        self.navigator = AutonomousNavigator()
    
    def process_sensor_data(self, camera_data, lidar_data, imu_data):
        """Process and fuse sensor data"""
        
        # Computer vision processing
        detected_objects = self.vision_system.detect_objects(camera_data)
        
        # Sensor fusion
        state_estimate = self.sensor_fusion.fuse_sensor_data(
            camera_data, lidar_data, imu_data
        )
        
        # Update navigation
        self.navigator.update_slam()
        
        return {
            'detected_objects': detected_objects,
            'state_estimate': state_estimate,
            'current_pose': self.navigator.current_pose
        }
    
    def plan_and_execute_motion(self, target_pose):
        """Plan and execute motion to target"""
        
        # Set target
        self.navigator.target_pose = target_pose
        
        # Navigate to target
        success = self.navigator.navigate_to_target()
        
        return success
```

### Key Considerations

1. **Real-time Performance**
   - Low-latency sensor processing
   - Fast inference for decision making
   - Efficient path planning algorithms
   - Real-time control loops

2. **Safety and Reliability**
   - Fail-safe mechanisms
   - Collision avoidance
   - Emergency stop procedures
   - Redundant sensor systems

3. **Environmental Adaptation**
   - Dynamic obstacle avoidance
   - Changing environment handling
   - Weather and lighting adaptation
   - Multi-robot coordination

4. **Energy Efficiency**
   - Power-aware algorithms
   - Battery management
   - Efficient sensor usage
   - Optimized motion planning

This comprehensive guide covers the essential aspects of machine learning in robotics, from computer vision to autonomous navigation and sensor fusion. 