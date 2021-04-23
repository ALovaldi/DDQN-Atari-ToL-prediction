# -*- coding: utf-8 -*-
# File: atari.py
# Author: Yuxin Wu

import numpy as np
import os
import threading
import cv2
import gym
import six
import argparse
from datetime import datetime, date, time
from atari_py.ale_python_interface import ALEInterface
from gym import spaces
from gym.envs.atari.atari_env import ACTION_MEANING

from skimage import metrics 
from scipy.stats import entropy

from tensorpack.utils import logger, execute_only_once, get_rng
from tensorpack.utils.fs import get_dataset_path

__all__ = ['AtariPlayer']

ROM_URL = "https://github.com/openai/atari-py/tree/gdb/atari_py/atari_roms"
_ALE_LOCK = threading.Lock()

eva = []

class Episode:
    def __init__(self, n_step, t_bool, time, cs):
        self.n_step = n_step 
        self.t_bool = t_bool 
        self.time = time
        self.cs30 = cs
        self.cs60 = cs
        self.cs120 = cs
        self.cs240 = cs

class AtariPlayer(gym.Env):
    """
    A wrapper for ALE emulator, with configurations to mimic DeepMind DQN settings.

    Info:
        score: the accumulated reward in the current game
        gameOver: True when the current game is Over
    """

    def __init__(self, rom_file, viz=0,
                 frame_skip=0, nullop_start=30,
                 live_lost_as_eoe=True, max_num_frames=0,
                 grayscale=True):
        """
        Args:
            rom_file: path to the rom
            frame_skip: skip every k frames and repeat the action
            viz: visualization to be done.
                Set to 0 to disable.
                Set to a positive number to be the delay between frames to show.
                Set to a string to be a directory to store frames.
            nullop_start: start with random number of null ops.
            live_losts_as_eoe: consider lost of lives as end of episode. Useful for training.
            max_num_frames: maximum number of frames per episode.
            grayscale (bool): if True, return 2D image. Otherwise return HWC image.
        """
        super(AtariPlayer, self).__init__()
    
        if not os.path.isfile(rom_file) and '/' not in rom_file:
            rom_file = get_dataset_path('atari_rom', rom_file)
        assert os.path.isfile(rom_file), \
            "ROM {} not found. Please download at {}".format(rom_file, ROM_URL)

        try:
            ALEInterface.setLoggerMode(ALEInterface.Logger.Error)
        except AttributeError:
            if execute_only_once():
                logger.warn("You're not using latest ALE")

        # avoid simulator bugs: https://github.com/mgbellemare/Arcade-Learning-Environment/issues/86
        with _ALE_LOCK:
            self.ale = ALEInterface()
            self.rng = get_rng(self)
            self.ale.setInt(b"random_seed", 14)
            self.ale.setInt(b"max_num_frames_per_episode", max_num_frames)
            self.ale.setBool(b"showinfo", False)

            self.ale.setInt(b"frame_skip", 1)
            self.ale.setBool(b'color_averaging', False)
            # manual.pdf suggests otherwise.
            self.ale.setFloat(b'repeat_action_probability', 0.0)

            # viz setup
            if isinstance(viz, six.string_types):
                assert os.path.isdir(viz), viz
                self.ale.setString(b'record_screen_dir', viz)
                viz = 0
            if isinstance(viz, int):
                viz = float(viz)
            self.viz = viz
            if self.viz and isinstance(self.viz, float):
                self.windowname = os.path.basename(rom_file)
                cv2.namedWindow(self.windowname)

            self.ale.loadROM(rom_file.encode('utf-8'))
        self.width, self.height = self.ale.getScreenDims()
        self.actions = self.ale.getMinimalActionSet()

        self.live_lost_as_eoe = live_lost_as_eoe
        self.frame_skip = frame_skip
        self.nullop_start = nullop_start

        self.action_space = spaces.Discrete(len(self.actions))
        self.grayscale = grayscale
        shape = (self.height, self.width) if grayscale else (self.height, self.width, 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8)
        self._restart_episode()

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self.actions]

    def _grab_raw_image(self):
        """
        :returns: the current 3-channel image
        """
        m = self.ale.getScreenRGB()
        return m.reshape((self.height, self.width, 3))

    def _current_state(self):
        """
        :returns: a gray-scale (h, w) uint8 image
        """
        ret = self._grab_raw_image()
        # max-pooled over the last screen
        ret = np.maximum(ret, self.last_raw_screen)
        if self.viz:
            if isinstance(self.viz, float):
                cv2.imshow(self.windowname, ret)
                cv2.waitKey(int(self.viz * 1000))
        if self.grayscale:
            # 0.299,0.587.0.114. same as rgb2y in torch/image
            ret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
        return ret.astype('uint8')  # to save some memory

    def _restart_episode(self):
        with _ALE_LOCK:
            self.ale.reset_game()

        # random null-ops start
        n = self.rng.randint(self.nullop_start)
        self.last_raw_screen = self._grab_raw_image()
        for k in range(n):
            if k == n - 1:
                self.last_raw_screen = self._grab_raw_image()
            self.ale.act(0)

    def reset(self):
        if self.ale.game_over():
            self._restart_episode()
        return self._current_state()

    def render(self, *args, **kwargs):
        pass  # visualization for this env is through the viz= argument when creating the player

    def step(self, act):
        oldlives = self.ale.lives()
        r = 0
        for k in range(self.frame_skip):
            if k == self.frame_skip - 1:
                self.last_raw_screen = self._grab_raw_image()
            new_r = self.ale.act(self.actions[act])   
            r += new_r
            newlives = self.ale.lives()
            if self.ale.game_over() or (self.live_lost_as_eoe and newlives < oldlives):
                break

        isOver = self.ale.game_over()
        if self.live_lost_as_eoe:
            isOver = isOver or newlives < oldlives

        info = {'ale.lives': newlives}
        return self._current_state(), r, isOver, info

    def step_eval(self, n_ep, new_r, act):
        if len(eva) <= n_ep:
            obj = Episode(0, False, datetime.now(), self._current_state())
            eva.append(obj)
        
        eva[n_ep-1].n_step += 1
        # with open("step_{}.txt".format(n_ep), "a") as f:
        #   f.write("{}\n".format(eva[n_ep-1].n_step))
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
        parser.add_argument('--load', help='load model')
        parser.add_argument('--task', help='task to perform',choices=['play', 'eval', 'train'], default='train')
        parser.add_argument('--env', required=True, help='either an atari rom file (that ends with .bin) or a gym atari environment name')
        parser.add_argument('--algo', help='algorithm', choices=['DQN', 'Double', 'Dueling'], default='Double')
        parser.add_argument('--num-eval', default=50, type=int)
        parser.add_argument('--ntrain', default='0')
        args = parser.parse_args()
        if args.task == 'eval':
            game = args.env.split('.')[0] + "-" + args.ntrain
            folder = args.load.split('/')[3].split('.')[0]
            # game = "."
            # folder = "seaquest"
            if new_r != 0:
                t = datetime.now() - eva[n_ep-1].time
                with open("eval/{}/{}/reward_log_{}.txt".format(game, folder, n_ep), "a") as f: 
                    f.write("{}\t{}\t{}\t{}\n".format(t, eva[n_ep-1].n_step, new_r, act))
            if eva[n_ep-1].n_step % 30 == 0:
                m = np.sum((self._current_state().astype("float") - eva[n_ep-1].cs30.astype("float")) ** 2)
                m  /= float(eva[n_ep-1].cs30.shape[0] * eva[n_ep-1].cs30.shape[1])
                s = metrics.structural_similarity(eva[n_ep-1].cs30, self._current_state(), multichannel=False)
                eva[n_ep-1].cs30 = self._current_state()
                with open("eval/{}/{}/image_log_{}.txt".format(game, folder , n_ep), "a") as f: 
                  f.write("{}\t{}\t{}".format(eva[n_ep-1].n_step, m, s))
                  if eva[n_ep-1].n_step % 60 == 0:
                      m = np.sum((self._current_state().astype("float") - eva[n_ep-1].cs60.astype("float")) ** 2)
                      m  /= float(eva[n_ep-1].cs60.shape[0] * eva[n_ep-1].cs60.shape[1])
                      s = metrics.structural_similarity(eva[n_ep-1].cs60, self._current_state(), multichannel=False)
                      eva[n_ep-1].cs60 = self._current_state()
                      f.write("\t{}\t{}".format(m, s))
                      if eva[n_ep-1].n_step % 120 == 0:
                          m = np.sum((self._current_state().astype("float") - eva[n_ep-1].cs120.astype("float")) ** 2)
                          m  /= float(eva[n_ep-1].cs120.shape[0] * eva[n_ep-1].cs120.shape[1])
                          s = metrics.structural_similarity(eva[n_ep-1].cs120, self._current_state(), multichannel=False)
                          eva[n_ep-1].cs120 = self._current_state()
                          f.write("\t{}\t{}".format(m, s))
                          if eva[n_ep-1].n_step % 240 == 0:
                              m = np.sum((self._current_state().astype("float") - eva[n_ep-1].cs240.astype("float")) ** 2)
                              m  /= float(eva[n_ep-1].cs240.shape[0] * eva[n_ep-1].cs240.shape[1])
                              s = metrics.structural_similarity(eva[n_ep-1].cs240, self._current_state(), multichannel=False)
                              eva[n_ep-1].cs240 = self._current_state()
                              f.write("\t{}\t{}".format(m, s))
                  f.write("\n")  
                        
       

if __name__ == '__main__':
    import sys

    a = AtariPlayer(sys.argv[1], viz=0.03)
    num = a.action_space.n
    rng = get_rng(num)
    while True:
        act = rng.choice(range(num))
        state, reward, isOver, info = a.step(act)
        if isOver:
            print(info)
            a.reset()
        print("Reward:", reward)

   
