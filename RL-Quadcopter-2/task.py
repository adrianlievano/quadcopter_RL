import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, target_ang_pos = None, target_v = None, target_ang_v = None, ang_pos_tol = .1):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
            target_ang_pos: target/goal euler angle positions for the agent
            target_v = target/goal (Vx, Vy, Vz) for the agent
            target_ang_v = target/goal angular velocities for the agent
            pos_tol = tolerance in the distance between (x, y, z) and target (x, y, z) for the agent 
            ang_pos_tol = tolerance in the difference between euler angel positions and target positions for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.target_ang_pos = target_ang_pos if target_pos is not None else np.array([0.0, 0.0, 0.0]) 
        self.target_v = target_v if target_v is not None else np.array([0.0, 0.0, 0.0])
        self.target_ang_v = target_ang_v if target_ang_v is not None else np.array([0.0, 0.0, 0.0])
        
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        err = np.linalg.norm(self.sim.pose[:3] - self.target_pos)     

        
        reward = 1.0 - 0.3*err
        ## Case: Agent is far away from the target 
        
                # be not moving up/down/left/right/forward/backwards
        speed = abs(self.sim.v).sum()
        vel_tolerance = 0.01
        pos_tol = 0.1
        
        # be not spinning
        ang_speed = abs(self.sim.angular_v).sum()
        ang_vel_tolerance = 0.01
        
        if err <= pos_tol: # reward for being within tolerance near the goal
            if speed <= vel_tolerance and ang_speed <= ang_vel_tolerance: # reward for being broadly stationary
                # quad is near goal and broadly stationary
                reward = 10.0
            else:
                # reduce reward proportional to speed/ang_speed
                reward = 10.0 - 0.2 * speed - 0.2 * ang_speed
        else: # the quad is not near enough to the goal therefore it should be moving
            # reward based solely on distance at this point
            reward = -1.0 * distance
        
        ## Case: Agent is closer to the target & should stay stationary but still flying
        '''if (err <= .1):
            if (
            reward = 10
            
        elif (err <= 0.1)
            reward = 100
        else:
           ## Case: Agent is far away from the target and we penalize it proportionally to the error of the distance ##
            reward = -1.0*err
            #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()'''

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state