import gym
import numpy as np
from gfootball.env.wrappers import Simple115StateWrapper, CheckpointRewardWrapper


class ObservationWrapper(Simple115StateWrapper):
    """A wrapper that converts an observation to 115-features state."""

    def __init__(self, env):
        Simple115StateWrapper.__init__(self, env, fixed_positions=True)

    def observation(self, observation):
        """Converts an observation into simple115v2 format while reserves the raw observation.

        Args:
            Raw observation

        Returns:
            A simple115v2-format observation (for learning)
            Raw observation (for statistics)
        """
        return self.convert_observation(observation, self._fixed_positions), observation


class ShootRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a shooting reward."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._action = None
        self._ball_threshold_x = 0.66
        self._ball_threshold_y = 0.25
        self._shooting_reward = 0.2

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._action = action
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        if type(reward) is np.float32:
            reward = np.array([reward])

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the active player has the ball.
            if ('ball_owned_team' not in o or
                    o['ball_owned_team'] != 0 or
                    'ball_owned_player' not in o or
                    o['ball_owned_player'] != o['active']):
                continue

            # Computing the shooting reward.
            if (self._action[rew_index] == 12 and
                    o['ball'][0] >= self._ball_threshold_x and
                    abs(o['ball'][1]) < self._ball_threshold_y):
                reward[rew_index] += self._shooting_reward
        return reward


class PassRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a passing reward."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._action = None
        self._passing_reward = 0.1
        self._raw_observation_on_pass = None
        self._last_pass_player = None
        self._current_raw_observation = None
        self.count = 0

    def step(self, action):
        # Get the raw observation when some player in self-team controls the ball and does the pass action
        for agent_index in range(len(action)):
            if self._current_raw_observation[agent_index]['ball_owned_team'] == 0 and \
                    self._current_raw_observation[agent_index]['ball_owned_player'] == agent_index and \
                    action[agent_index] in [9, 10, 11]:
                self._raw_observation_on_pass = self._current_raw_observation
                self._last_pass_player = agent_index
                break
        observation, reward, done, info = self.env.step(action)
        _, self._current_raw_observation = observation
        self._action = action
        return observation, self.reward(reward), done, info

    def reset(self, **kwargs):
        self._raw_observation_on_pass = None
        self._last_pass_player = None
        (obs_list, cur_raw_obs) = self.env.reset(**kwargs)
        self._current_raw_observation = cur_raw_obs
        self.count = 0
        return obs_list, cur_raw_obs

    def reward(self, reward):
        if type(reward) is np.float32:
            reward = np.array([reward])

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        assert len(reward) == len(observation)

        if self.count < 10:
            # ball is get by someone
            if self._last_pass_player and self._current_raw_observation[0]['ball_owned_team'] != -1:
                # pass failure
                if self._current_raw_observation[0]['ball_owned_team'] != 0:
                    # failure
                    reward[self._last_pass_player] -= self._passing_reward
                    # print("Player {} passed and failed ".format(self._last_pass_player))
                # pass success
                if self._current_raw_observation[0]['ball_owned_team'] == 0:
                    reward[self._last_pass_player] += self._passing_reward
                    self.count += 1
                    # print("Player {} passed and succeed ".format(self._last_pass_player))
                # reset internal variables when finishing computing reward
                self._raw_observation_on_pass = None
                self._last_pass_player = None

        return reward


import gym
import numpy as np
from gfootball.env import observation_preprocessing
from gfootball.env.wrappers import FrameStack


class VectorWrapper(gym.ObservationWrapper):
    """A wrapper that converts an observation."""

    def __init__(self, env, total_agents, fixed_positions=True, drop_invalid=False):
        """Initializes the wrapper.

        Args:
        env: an envorinment to wrap
        fixed_positions: whether to fix observation indexes corresponding to teams
        drop_invalid: drop the dimensions of absent players that filled by -1 (when not in 11 vs 11)
        """
        gym.ObservationWrapper.__init__(self, env)
        self.drop_invalid = drop_invalid
        if not self.drop_invalid:  # original simple115
            shape = (115,)
        else:
            self.obs_dim = 5 * 8 + 21  # 5_vs_5 scenario
            shape = (self.obs_dim,)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        self._fixed_positions = fixed_positions

    def observation(self, observation):
        """Converts an observation.
        
        Args:
        observation: observation that the environment returns

        Returns:
        converted observation (for learning)
        and original observation (for statistics)
        """
        return self.convert_observation(observation, self._fixed_positions), observation

    def convert_observation(self, observation, fixed_positions):
        """Converts an observation into array format.

        Args:
        observation: observation that the environment returns
        fixed_positions: Players and positions are always occupying 88 fields
                        (even if the game is played 1v1).
                        If True, the position of the player will be the same - no
                        matter how many players are on the field:
                        (so first 11 pairs will belong to the first team, even
                        if it has less players).
                        If False, then the position of players from team2
                        will depend on number of players in team1).

        Returns:
        [N, N*2(teams)*2(positions&directions)*2(x&y)+27] shaped representation
        where N stands for the number of players being controlled.
        """

        def do_flatten(obj):
            """Run flatten on either python list or numpy array."""
            if type(obj) == list:
                return np.array(obj).flatten()
            return obj.flatten()

        final_obs = []
        for obs in observation:
            o = []
            if fixed_positions:
                for i, name in enumerate(['left_team', 'left_team_direction',
                                        'right_team', 'right_team_direction']):
                    o.extend(do_flatten(obs[name]))
                    # # If there were less than 11vs11 players
                    # # `simple115` backfill missing values with -1.
                    if not self.drop_invalid:
                        if len(o) < (i + 1) * 22:
                            o.extend([-1] * ((i + 1) * 22 - len(o)))
            else:
                o.extend(do_flatten(obs['left_team']))
                o.extend(do_flatten(obs['left_team_direction']))
                o.extend(do_flatten(obs['right_team']))
                o.extend(do_flatten(obs['right_team_direction']))

          # 88 = 11 (players) * 2 (teams) * 2 (positions & directions) * 2 (x & y)
            if not self.drop_invalid:
                if len(o) < 88:
                    o.extend([-1] * (88 - len(o)))

            # ball position
            o.extend(obs['ball'])
            # ball direction
            o.extend(obs['ball_direction'])
            # one hot encoding of which team owns the ball
            if obs['ball_owned_team'] == -1:
                o.extend([1, 0, 0])
            if obs['ball_owned_team'] == 0:
                o.extend([0, 1, 0])
            if obs['ball_owned_team'] == 1:
                o.extend([0, 0, 1])

            if not self.drop_invalid:
                active = [0] * 11
            else:
                active = [0] * 5  # in the 5_vs_5 case
                
            if obs['active'] != -1:
                active[obs['active']] = 1
            o.extend(active)

            game_mode = [0] * 7
            game_mode[obs['game_mode']] = 1
            o.extend(game_mode)
            final_obs.append(o)
        return np.array(final_obs, dtype=np.float32)


class SMMWrapper(gym.ObservationWrapper):
  """A wrapper that convers observations into a minimap format."""

  def __init__(self, env,
               channel_dimensions=(observation_preprocessing.SMM_WIDTH,
                                   observation_preprocessing.SMM_HEIGHT)):
    gym.ObservationWrapper.__init__(self, env)
    self._channel_dimensions = channel_dimensions
    action_shape = np.shape(self.env.action_space)
    shape = (action_shape[0] if len(action_shape) else 1, channel_dimensions[0],
             channel_dimensions[1],
             len(
                 observation_preprocessing.get_smm_layers(
                     self.env.unwrapped._config)))
    self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=shape, dtype=np.uint8)

  def observation(self, obs):
    """Returns a list of minimap observations given the raw features for each
    active player.

    Args:
        observation: raw features from the environment
        config: environment config
        channel_dimensions: resolution of SMM to generate

    Returns:
        (N, W, H, C) - shaped np array representing SMM. N stands for the number of
        players we are controlling.
  """
    return np.swapaxes(observation_preprocessing.generate_smm(
        obs, channel_dimensions=self._channel_dimensions,
        config=self.env.unwrapped._config), 1, 2), obs
