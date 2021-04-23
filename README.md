# DDQN-Atari-ToL-prediction

## Usage:
Follow the indication of the README in DQN and DQN-eval to train and evaluate the model.

###Training
The folder DQN contains the files to train the model. The original model can be found at: https://github.com/tensorpack/tensorpack/tree/master/examples/DeepQNetwork
Added option '--ntrain n' to modify the output folder to 'train_log/DQN-GameName-n'.
A model is saved every 4 epochs.
This make possible to train multiple models on the same game in parallel.

Start Training:
```
./DQN.py --env breakout.bin --task train --ntrain 0
# use `--algo` to select other DQN algorithms. See `-h` for more options.
# use '--ntrain' to train the same game in parallel. Output folder change into train_log/DQN-Breakout-ntrain
```

### Evalutaion
The folder DQN-eval contains the files to evalute the model. The original model can be found at: https://github.com/tensorpack/tensorpack/tree/master/examples/DeepQNetwork
Function to collect statistics of the training are been added.
Before starting the evalution create a folder named 'eval/gameName-ntrain/model-x'
Where:
- "gameName-ntrain" is the name of the model to evaluate, if the option '--ntrian' is not used in the training the folder name is just "gameName" 
- model-x is the name of the model to be evaluated 

In the folder created will be created the files storing the statitics.
- image_log_n.txt -> tsv file of evalation game n contating information on the MSE and SSIM computed between frames 30, 60, 120, and 240 frames apart
- reward_log.txt ->  tsv file of evalation game n contating information on the time, frame, reward value, and action of reward with value != 0 

Evaluation of 50 episodes:
```
./DQN.py --env breakout.bin --task eval --load ./train_log/DQN-Breakout-0/model-12500000.index"
```

### Data analisys
DQN_eval.ipynb is the Jupyter Notebook contating the panda fucntion to analyze the data and produce the graphs.


