# Final Project

This repository is my solution of the final project. This documentation is an instructions to run this code and reproduce the results.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

The requirements is the same as the original repo.

## Training

To train models, run this command:

```train
cd code
python main.py hard_0 sarsa 1000
```
`hard_0` is the name of map. `sarsa` is the basic TD algorithm. You can also use `qlearning`. `1000` is the training episodes.

There are some hyperparameters you can modify in the line 10-14 of `main.py`, includes bias, epsilon_decay, N, gamma.

Training data will be saved in the folder `results/bias_{bias}_epsilon_decay_{epsilon_decay}_N_{N}`. The model is saved as `.pkl` file.

## Test

To test a model, run:

```test
python main.py hard_0 sarsa 10 hard_0_sarsa.pkl
```

`hard_0` is the name of map. `sarsa` is the basic TD algorithm. `10` is the test episodes. `hard_0_sarsa.pkl` is the file of model. It needs to save in `results/`.

Test results is saved in the folder `results/test_{model_file_name}`.

## Some information to reproduce my results

My models are saved in the file `results/`. To help reproduce the results, there is some information about the models:

| model file         | map  | bias | $\epsilon \_dacay$ | N | TD algor. | training episodes | avg test returns |
| ------------------ | ---------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
|  hard_0_sarsa.pkl  |     hard_0         |      0.005       | 0.0001 | 200 | SARSA |1000 |9.57|
|   hard_0_qlearning.pkl |     hard_0         |      0.005       | 0.0001 | 200 | QLearning |1000 |9.36|
|  hard_1_sarsa.pkl |     hard_1         |      0.005       | 0.0001 | 200 | SARSA |1300 |9.50|
|  hard_1_qlearning.pkl  |     hard_1         |      0.005       | 0.0001 | 200 | QLearning |1300 |9.57|

To test my models quickly, you can run:
```test
cd code
python main.py hard_0 sarsa 10 hard_0_sarsa.pkl
python main.py hard_0 qlearning 10 hard_0_qlearning.pkl
python main.py hard_1 sarsa 10 hard_1_sarsa.pkl
python main.py hard_1 qlearning 10 hard_1_qlearning.pkl
```

## Hardware and random seed
I use MacBook Pro to conduct the  experiment. The CPU is M1 Pro. The memory is 16GB. The OS is macOS Monterey v12.5. If your device can run the original repository, then the device is suitable to run my code.

Compared to the original code, my solution doesn't have a new random seed.

## About code:  important files and comments
The important parts about my solution are mainly in `main.py` and `agent.py`. And I comment those parts by ## line.

## Contributing

This repository is based on `thomyphan/autonomous-decision-making`.
