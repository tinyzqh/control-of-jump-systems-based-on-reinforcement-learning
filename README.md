
# *强化学习用于PID的参数整定和电流补偿控制。*

# Introduce

This is the implementation of the paper “Control Strategy of Speed Servo Systems Based on Deep Reinforcement Learning”.

For more paper information, please checkout the the paper [Link](https://www.mdpi.com/1999-4893/11/5/65).


# Class PID

The class PID algorithms has no training process.

```bash
python run.py --choose_model class_pid --curve_type trapezoidal --height 1000 --run_type test
```

Plot the result:

```bash
python plot_result.py --choose_model class_pid --curve_type trapezoidal --height 1000 --run_type test
```

<div align=center>
    <span class='gp-n'>
        <img src='https://github.com/tinyzqh/control-of-jump-systems-based-on-reinforcement-learning/blob/main/results/ChooseModel_class_pid_CurveType_trapezoidal_Height_1000_DumpSystem_False_RunType_test/ecValues.png' width="250" alt="ecValues"/>
        <img src='https://github.com/tinyzqh/control-of-jump-systems-based-on-reinforcement-learning/blob/main/results/ChooseModel_class_pid_CurveType_trapezoidal_Height_1000_DumpSystem_False_RunType_test/iaeValues.png' width="250" alt="iaeValues"/>
        <img src='https://github.com/tinyzqh/control-of-jump-systems-based-on-reinforcement-learning/blob/main/results/ChooseModel_class_pid_CurveType_trapezoidal_Height_1000_DumpSystem_False_RunType_test/radValues.png' width="250" alt="radValues"/>
    </span>
</div>


# Parameters Adjustment Of PID

Search the parameter of PID Based DDPG Algorithm.

```bash
python run.py --choose_model search_pid_parameter --curve_type trapezoidal --height 1000 --run_type train
```

experimental operation process:

<div align=center>
    <span class='gp-n'>
        <img src='https://github.com/tinyzqh/control-of-jump-systems-based-on-reinforcement-learning/blob/main/results/ChooseModel_search_pid_parameter_CurveType_trapezoidal_Height_1000_DumpSystem_False_RunType_train/reward.PNG' width="500" alt="epRewards_fig"/>
    </span>
</div>


# Electric Current Compensation Of RL-PID


```bash
python run.py --choose_model search_electric --curve_type trapezoidal --height 1000 --run_type train
```

test


```bash
python run.py --choose_model search_electric --curve_type trapezoidal --height 1000 --run_type test
```

Plot the result:

```bash
python plot_result.py --choose_model search_electric --curve_type trapezoidal --height 1000 --run_type train
```

the result show:

<div align=center>
    <span class='gp-n'>
        <img src='https://github.com/tinyzqh/control-of-jump-systems-based-on-reinforcement-learning/blob/main/results/ChooseModel_search_electric_CurveType_trapezoidal_Height_1000_DumpSystem_False_RunType_train/epRewards.png' width="500" alt="epRewards"/>
    </span>
</div>


# Dependencies

The code was tested under Ubuntu 16 and uses these packages:

- tensorflow-gpu==1.14.0
- atari-py==0.2.6
- gym==0.17.3
- numpy==1.91.3

more packages described in requirements.txt


Citing
------
If you find this open source release useful, please reference in your paper:


> Chen P, He Z, Chen C, et al. (2018).
> Control strategy of speed servo systems based on deep reinforcement learning[J].
> *Algorithms, 2018, 11(5): 65.*.
