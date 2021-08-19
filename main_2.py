import numpy as np
import sys
import matplotlib.pyplot as plt
from sko.GA import GA
from math import exp

def my_function(m1, m2, m3, de1, de2, de3, p1, p2, p3, q1, q2, q3):
    mf1 = []
    mf2 = []
    mf3 = []
    x = []
    y = []
    for i in range(1, 120):
        x.append(i / 10)
        mf1.append( exp( (-(x[i - 1] - m1) ** 2) / ( 2 * de1 ** 2)  ) )
        mf2.append( exp( (-(x[i - 1] - m2) ** 2) / ( 2 * de2 ** 2)  ) )
        mf3.append( exp( (-(x[i - 1] - m3) ** 2) / ( 2 * de3 ** 2)  ) )

        b = mf1[i - 1] + mf2[i - 1] + mf3[i - 1]
        a1 = mf1[i - 1] * ( p1 * x[i - 1] + q1 )
        a2 = mf2[i - 1] * ( p2 * x[i - 1] + q2 )
        a3 = mf3[i - 1] * ( p3 * x[i - 1] + q3 )

        a = a1 + a2 +a3
        y.append(a / b)

    return [x, y]


#### START PRECIPITACION
m1 = 0
de1 = 2.5

m2 = 5
de2 = 2

m3 = 12
de3 = 2

p1 = 1
q1 = 20

p2 = 3.8
q2 = 10

p3 = 1
q3 = 10

# Precipitacion
x1, y1 = my_function(m1, m2, m3, de1, de2, de3, p1, p2, p3, q1, q2, q3)

#### END PRECIPITACION

### START TEMPERATURA
m1 = 3
de1 = 4

m2 = 6.8
de2 = 2

m3 = 12
de3 = 3

p1 = 4.9
q1 = 23

p2 = 2
q2 = -0

p3 = 2
q3 = -4
#Temperatura
x2, y2 = my_function(m1, m2, m3, de1, de2, de3, p1, p2, p3, q1, q2, q3)

fig = plt.figure()
fig.set_size_inches(18.5, 10.5)

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)


ax1.title.set_text('Promedio de lluvias y temperatura entre 1971 a 1998')
ax1.plot( x1, y1, color='blue', linestyle='dashed', label='Precipitacion' )
ax1.plot( x2, y2, color='red', linestyle='dashed', label='Temperatura' )
ax1.legend(loc="upper right")
ax1.grid()


## GA HERE TO FIGURE OUT OUR NETWORK NEURONAL


def my_function_ga(m1, m2, m3, de1, de2, de3, p1, p2, p3, q1, q2, q3):
    mf1 = []
    mf2 = []
    mf3 = []
    x = []
    y = []
    for i in range(1, 120):
        x.append(i / 10)
        mf1.append( exp( (-(x[i - 1] - m1) ** 2) / ( 2 * de1 ** 2)  ) )
        mf2.append( exp( (-(x[i - 1] - m2) ** 2) / ( 2 * de2 ** 2)  ) )
        mf3.append( exp( (-(x[i - 1] - m3) ** 2) / ( 2 * de3 ** 2)  ) )

        b = mf1[i - 1] + mf2[i - 1] + mf3[i - 1]
        a1 = mf1[i - 1] * ( p1 * x[i - 1] + q1 )
        a2 = mf2[i - 1] * ( p2 * x[i - 1] + q2 )
        a3 = mf3[i - 1] * ( p3 * x[i - 1] + q3 )

        a = a1 + a2 +a3
        y.append(a / b)

    return y

def ga_fun(p):
    a, b, c, d, e, f, g, h, i, j, k, m = p
    residuals = np.float64(
            my_function_ga(a, b, c, d, e, f, g, h, i, j, k, m)
        ).sum()
    # residuals = my_function_ga(a, b, c, d, e, f, g, h, i, j, k, m)


    #residuals = np.float64(abs( - y1)).sum()
    return residuals


generations = 12
ga = GA(func=ga_fun, n_dim=12, size_pop=136, max_iter=generations, prob_mut=0.01,
        lb=0, ub=21)


best_params, residuals = ga.run()
print('best_x:', best_params, '\n', 'best_y:', residuals)


y_predict = my_function_ga(*best_params)
print(y_predict)
best_y_generation = ga.generation_best_Y
best_x_generation = ga.generation_best_X

#print(best_x_generation)

ax2.plot( x1, y_predict, color='blue', linestyle='dashed', label='Precipitacion' )

#plt.margins(0)
#plt.rcParams['axes.xmargin'] = 0
#plt.rcParams['axes.ymargin'] = 0
#plt.show()