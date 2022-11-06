<!--
 * @Author: myyao
 * @Date: 2022-02-02 13:01:13
 * @Description: 
-->


# Summary

This folder is used to measure the energy consumption of different modes. 
To run this file, you should in `motivation` folder, and then,
```sh
python main.py
```

In final, you can see the information of TABLE shown in paper.

The output when running main.py is,
```
relay num: 4
__________________________________________________
local computing:  0.0001404956098700493
direct: [0.00022395467425314646, 0.00023108995282211444]
direct power: [0.1, 0.1]
**************************************************
This is the result of TABLE 1: 'An example for energy efficiency computation'
**************************************************
cc: [0.00010504167104346851, 0.00010523242757622529]
df: [0.00010436568110534927, 0.00010710632237225786]
cc power: [0.1, 0.1]
df mobile power: [0.1, 0.1]
df relay power: [0.01, 0.01]
```