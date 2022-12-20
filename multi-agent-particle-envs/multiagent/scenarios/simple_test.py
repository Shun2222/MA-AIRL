import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx   
import json
import configparser
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self):
        self.env_info = EnvInfo()
        self.entities_color = None
        self.agent_id = 1

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 1
        num_goals = 1
        num_obstacles = self.env_info.num_obstacles
        world.collaborative = True
        radius = 2/(2*np.max(self.env_info.state_size))
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.id = self.agent_id
            agent.collide = True
            agent.silent = True
            agent.size = radius*0.8
            #agent.accel = 3.0 
            #agent.max_speed = 1.0 
        # add landmarks
        world.goals = [Landmark() for i in range(num_goals)]
        for i, landmark in enumerate(world.goals):
            landmark.name = 'landmark %d' % i
            landmark.id = self.agent_id
            landmark.collide = False
            landmark.movable = False
            landmark.size = radius*0.5
        # add obstacles
        world.obstacles = [Landmark() for i in range(num_obstacles)]
        for i, landmark in enumerate(world.obstacles):
            landmark.name = 'obstacle %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = radius

        world.landmarks += world.goals
        world.landmarks += world.obstacles
        # make initial conditions
        self.entities_color = EntitiesColor(world)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        starts_pos = self.env_info.starts_pos
        goals_pos = self.env_info.goals_pos
        obstacles_pos = self.env_info.obstacles_pos
        
        # properties for agents
        for i, agent in enumerate(world.agents):
            colorVal = self.entities_color.get_color(i, alpha=0.4)
            agent.color = np.array(colorVal)
        # properties for landmarks
        for i, landmark in enumerate(world.goals):
            colorVal = self.entities_color.get_color(i, alpha=0.8)
            landmark.color = np.array(colorVal)
        # random properties for landmarks
        for i, obstacle in enumerate(world.obstacles):
            obstacle.color = np.array((0,0,0,1))
        
        # set initial states
        for i, agent in enumerate(world.agents):
            agent.state.p_pos =  np.array(self.env_info.discrete_to_continue(starts_pos[agent.id]), dtype='float64')
            #agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.goals):
            landmark.state.p_pos = np.array(self.env_info.discrete_to_continue(goals_pos[agent.id]), dtype='float64')
            #landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, obstacle in enumerate(world.obstacles):
            obstacle.state.p_pos = np.array(self.env_info.discrete_to_continue(obstacles_pos[i]), dtype='float64')
            #obstacle.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            obstacle.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return np.minimum(dist - dist_min, 0.0)  # True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for i, a in enumerate(world.agents):
            if agent != a:
                continue
            #if self.is_collision(world.agents[i], world.landmarks[i]):
            #    rew += 10
            dist = np.sqrt(np.sum(np.square(a.state.p_pos - world.goals[i].state.p_pos)))
            rew -= dist
        if agent.collide:
            for a in world.agents:
                if a==agent:
                    continue
                if self.is_collision(a, agent):
                    rew -= 10
            for o in world.obstacles:
                if self.is_collision(o, agent):
                    rew -= 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        
        agent.color = self.entities_color.get_color(agent.id, alpha=0.4)
        for i, a in enumerate(world.agents):
            if agent==a: 
                if self.is_collision(agent, world.goals[i]):
                    agent.color = np.array((0,1,0,0.5))
                continue
            if self.is_collision(agent, a):
                agent.color = np.array((1,0,0,0.5))
        for obstacle in world.obstacles:
            if self.is_collision(agent, obstacle):
                agent.color = np.array((1,0,0,0.5))

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos])

    def done(self, agent, world):
        if world.time >= 50:
            return True
        else:
            return False

# dis = discrete, con = continue
class EnvInfo():
    def __init__(self):
        config_ini = configparser.ConfigParser()
        config_ini.optionxform = str
        config_ini.read('/MA-AIRL/multi-agent-particle-envs/multiagent/config/config.ini', encoding='utf-8')
        ENV = json.loads(config_ini.get("ENV", "ENV_INFO"))

        self.num_agents = int(config_ini.get(ENV, "N_AGENTS"))
        self.state_size = json.loads(config_ini.get(ENV, "STATE_SIZE")) 

        self.obstacles_pos = json.loads(config_ini.get(ENV, "OBSTACLE")) 
        self.num_obstacles = len(self.obstacles_pos)
        if len(self.obstacles_pos[0])==0:
            self.num_obstacles = 0
        
        self.starts_pos = []
        self.goals_pos = []
        for i in range(self.num_agents):
            agent_info = json.loads(config_ini.get(ENV,"AGENT_START_GOAL_EXPERT"+str(i+1)))
            self.starts_pos.append(agent_info[0][0])
            self.goals_pos.append(agent_info[0][1])

        self.num_row = self.state_size[0]
        self.num_col = self.state_size[1]

    def discrete_to_continue(self, row_col):
        # row is y, col is x -> [row, col] -> [x, y]=[f(col), f(row)] 
        return [-1+(1+2*row_col[1])/self.num_col, 1-(1+2*row_col[0])/self.num_row]
    def continue_to_discrete(self, pos):
        ## row is y, col is x -> [x, y] -> [row, col]=[f(x), f(y)] 
        return [self.num_row-1-int((pos[1]+1)/(2/self.num_row)), int((pos[0]+1)/(2/self.num_col))]

# color management
class EntitiesColor():
    def __init__(self, world):
        cmap = plt.cm.jet
        cNorm  = colors.Normalize(vmin=0, vmax=len(world.agents))
        self.scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)        

    def get_color(self, i, alpha=1.0):
        return np.array(self.scalarMap.to_rgba(i, alpha=alpha))
