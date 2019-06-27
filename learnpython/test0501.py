import math

## softMax 函数

z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
z_exp = [math.exp(i) for i in z]
print('z_exp', z_exp)

sum_z_exp = sum(z_exp)
print('sum_z_exp', sum_z_exp)

softmax = [round(i/sum_z_exp, 3) for i in z_exp]
print('softmax', softmax)
# softmax [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
