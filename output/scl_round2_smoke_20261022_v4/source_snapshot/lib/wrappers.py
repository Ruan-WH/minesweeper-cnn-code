
import gym
import numpy as np


def _reshape_observation(player_board):
    return player_board[np.newaxis, ...]


class CustomObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env, encoding_type="none"):
        super(CustomObservationWrapper, self).__init__(env)
        self.encoding_type = encoding_type
        if encoding_type == "none":
            # 修改observation_space的形状为(1, num_rows, num_cols)
            self.observation_space = gym.spaces.Box(low=-2, high=9, shape=(1, self.num_rows, self.num_cols),
                                                    dtype=np.float32)
        elif encoding_type == "one_hot":
            # 修改observation_space的形状,10个通道分别表示1-8, 0, 未知
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10, self.num_rows, self.num_cols),
                                                    dtype=np.float32)
            self.one_hot_template = np.eye(8, dtype=np.float32)
        elif encoding_type == "condensed":
            # 修改observation_space的形状, 2个通道分别表示1-8, 未知
            self.observation_space = gym.spaces.Box(low=0, high=8, shape=(2, self.num_rows, self.num_cols),
                                                    dtype=np.float32)
        else:
            raise ValueError("Invalid encoding_type. Supported types are 'one_hot' and 'condensed'")

    def observation(self, observation):
        player_board = np.array(observation, dtype=np.int32)
        if self.encoding_type == "none":
            return _reshape_observation(player_board)
        elif self.encoding_type == "one_hot":
            return self._one_hot_encode(player_board)
        elif self.encoding_type == "condensed":
            return self._condensed_encode(player_board)

    def _one_hot_encode(self, player_board):

        # 创建一个独热编码表示的 observation
        one_hot_obs = np.zeros((10, self.num_rows, self.num_cols), dtype=np.float32)

        # 填充1-8单元格的独热编码
        one_to_eight_cells = (player_board >= 1) & (player_board <= 8)
        one_hot_obs[:8, one_to_eight_cells] = self.one_hot_template[player_board[one_to_eight_cells] - 1].T

        # 填充0单元格的独热编码
        zero_cells = player_board == 0
        one_hot_obs[8, zero_cells] = 1.0

        # 填充未知单元格的独热编码
        unknown_cells = player_board == -1
        one_hot_obs[9, unknown_cells] = 1.0

        return one_hot_obs

    def _condensed_encode(self, player_board):

        # 创建一个压缩编码表示的 observation
        condensed_obs = np.zeros((2, self.num_rows, self.num_cols), dtype=np.float32)

        # 第一通道: 1-8的整数值
        one_to_eight_cells = (player_board >= 1) & (player_board <= 8)
        condensed_obs[0, one_to_eight_cells] = player_board[one_to_eight_cells]

        # 第二通道: 未知单元格为1, 其他为0
        unknown_cells = player_board == -1
        condensed_obs[1, unknown_cells] = 1.0

        return condensed_obs

    @property
    def reset_button(self):
        if hasattr(self.env, 'reset_button'):
            return self.env.reset_button
        else:
            raise AttributeError("The wrapped environment does not have a 'reset_button' attribute.")

    @property
    def board(self):
        if hasattr(self.env, 'board'):
            return self.env.board
        else:
            raise AttributeError("The wrapped environment does not have a 'board' attribute.")

    @property
    def done(self):
        if hasattr(self.env, 'done'):
            return self.env.done
        else:
            raise AttributeError("The wrapped environment does not have a 'done' attribute.")

    @property
    def num_rows(self):
        if hasattr(self.env, 'num_rows'):
            return self.env.num_rows
        else:
            raise AttributeError("The wrapped environment does not have a 'num_rows' attribute.")

    @num_rows.setter
    def num_rows(self, value):
        self.env.num_rows = value

    @property
    def num_cols(self):
        if hasattr(self.env, 'num_cols'):
            return self.env.num_cols
        else:
            raise AttributeError("The wrapped environment does not have a 'num_cols' attribute.")

    @num_cols.setter
    def num_cols(self, value):
        self.env.num_cols = value

    @property
    def button(self):
        if hasattr(self.env, 'button'):
            return self.env.button
        else:
            raise AttributeError("The wrapped environment does not have a 'button' attribute.")

    @button.setter
    def button(self, value):
        self.env.button = value

    @property
    def stats(self):
        if hasattr(self.env, 'stats'):
            return self.env.stats
        else:
            raise AttributeError("The wrapped environment does not have a 'stats' attribute.")

    @stats.setter
    def stats(self, value):
        self.env.stats = value


def make_env():
    pass
