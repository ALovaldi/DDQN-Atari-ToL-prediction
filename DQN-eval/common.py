# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu

import multiprocessing
import numpy as np
import random
import time
import argparse
from six.moves import queue

from tensorpack.callbacks import Callback
from tensorpack.utils import logger, get_tqdm
from tensorpack.utils.concurrency import ShareSessionThread, StoppableThread
from tensorpack.utils.stats import StatCounter
n_action = [0] *18

def play_one_episode(env, func, n_ep, render=False):
    def predict(s):
        """
        Map from observation to action, with 0.01 greedy.
        """
        s = np.expand_dims(s, 0)  # batch
        act = func(s)[0][0].argmax()
        if random.random() < 0.01:
            spc = env.action_space
            act = spc.sample()
        return act

    ob = env.reset()
    sum_r = 0
    while True:
        act = predict(ob)
        n_action[act] = n_action[act] + 1
        ob, r, isOver, info = env.step(act)
        env.step_eval(n_ep, r, act)
        if render:
            env.render()
        sum_r += r
        if isOver:
            return sum_r

def play_n_episodes(player, predfunc, nr, render=False):
    logger.info("Start Playing ... ")
    for k in range(nr):
        score = play_one_episode(player, predfunc, render=render)
        print("{}/{}, score={}".format(k, nr, score))

def eval_with_funcs(predictors, nr_eval, get_player_fn, verbose=False):
    """
    Args:
        predictors ([PredictorBase])
    """
    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue
            self.n_ep = 0

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player_fn(train=False)
                while not self.stopped():
                    try:
                        self.n_ep += 1
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
                          # game = "seaquest-0"
                          # folder = "model-12500000"
                          with open("eval/{}/{}/reward_log_{}.txt".format(game, folder, self.n_ep), "w") as f:
                            f.write("TIME\tSTEP\tREWARD\tACTION\n")
                          with open("eval/{}/{}/image_log_{}.txt".format(game, folder, self.n_ep), "w") as f:
                            f.write("STEP\tMSE30\tSSIM30\tMSE60\tSSIM60\tMSE120\tSSIM120\tMSE240\tSSIM240\n")
                          score = play_one_episode(player, self.func, self.n_ep)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, score)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predictors]
    while len(threads) != 1:
        threads.pop();
    logger.info("{} workers in threads".format(len(threads)))
    for k in threads: 

        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()

    def fetch():
        r = q.get()
        stat.feed(r)
        if verbose:
            logger.info("Score: {}".format(r))
        

    for i in get_tqdm(range(nr_eval)):
        fetch()
    # waiting is necessary, otherwise the estimated mean score is biased
    logger.info("Waiting for all the workers to finish the last run...")
    for k in threads:
        k.stop()
    for k in threads:
        k.join()
    while q.qsize():
        fetch()

    if stat.count > 0:
        return (stat.average, stat.max)
    return (0, 0)


def eval_model_multithread(pred, nr_eval, get_player_fn):
    """
    Args:
        pred (OfflinePredictor): state -> [#action]
    """
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    logger.info("NR_PROC {}".format(NR_PROC))
    with pred.sess.as_default():
        mean, max = eval_with_funcs(
            [pred] * NR_PROC, nr_eval,
            get_player_fn, verbose=True)
    logger.info("Average Score: {}; Max Score: {}".format(mean, max))
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform',choices=['play', 'eval', 'train'], default='train')
    parser.add_argument('--env', required=True, help='either an atari rom file (that ends with .bin) or a gym atari environment name')
    parser.add_argument('--algo', help='algorithm', choices=['DQN', 'Double', 'Dueling'], default='Double')
    parser.add_argument('--num-eval', default=50, type=int)
    parser.add_argument('--ntrain', default='0')
    args = parser.parse_args()
    with open("eval/{}_action.txt".format(args.env.split('.')[0]), "a") as f:
        f.write("{}".format(n_action))


class Evaluator(Callback):
    def __init__(self, nr_eval, input_names, output_names, get_player_fn):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * NR_PROC

    def _trigger(self):
        t = time.time()
        mean, max = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('mean_score', mean)
        self.trainer.monitors.put_scalar('max_score', max)

 
