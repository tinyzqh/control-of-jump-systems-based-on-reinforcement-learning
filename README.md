
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
        <img src='https://github.com/tinyzqh/Kagebunsin-no-jyutu/blob/master/figures/acrobots-swingupFigure_1.png' width="250" alt="多尺度acrobots-swingup"/>
        <img src='https://github.com/tinyzqh/Kagebunsin-no-jyutu/blob/master/figures/cartpole-balanceFigure_1.png' width="250" alt="多尺度cartpole-balance"/>
        <img src='https://github.com/tinyzqh/Kagebunsin-no-jyutu/blob/master/figures/hopper-hopFigure_1.png' width="250" alt="多尺度hopper-hop"/>
    </span>
</div>


# Parameters Adjustment Of PID

Search the parameter of PID Based DDPG Algorithm.




# Electric Current Compensation Of PID

