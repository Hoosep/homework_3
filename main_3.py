import numpy as np
import matplotlib.pyplot as plt
import time
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

#### START TEMPERATURA

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
###END TEMPERATURA

fig = plt.figure()
fig.set_size_inches(18.5, 8.5)

precipitacion_ax1 = fig.add_subplot(2, 2, 1)
temperatura_ax2 = fig.add_subplot(2, 2, 2)
error_ax = fig.add_subplot(2, 2, (3,4))

fig.suptitle('Promedio de lluvias y temperatura entre 1971 a 1998', fontsize=20)
precipitacion_ax1.plot( x1, y1, color='blue', linestyle='dashed', label='Precipitacion' )

temperatura_ax2.plot(x2, y2, color='red', linestyle='dashed', label='Temperatura' )

precipitacion_ax1.legend(loc="upper right")
precipitacion_ax1.grid()
## GA HERE TO FIGURE OUT OUR NETWORK NEURONAL

y_curve_fitting_real = np.array(y1)
x_curve_fitting_real = np.array(x1)

y_temperature_real = np.array(y2)
x_temperature_real = np.array(x2)
# print(y_curve_fitting_real)
#print("*"*80)
def my_function_ga(x, m1, m2, m3, de1, de2, de3, p1, p2, p3, q1, q2, q3):
    mf1 = []
    mf2 = []
    mf3 = []
    y = []
    index = 1
    for i in x:
        mf1.append( exp( (-(i - m1) ** 2) / ( 2 * de1 ** 2)  ) )
        mf2.append( exp( (-(i - m2) ** 2) / ( 2 * de2 ** 2)  ) )
        mf3.append( exp( (-(i - m3) ** 2) / ( 2 * de3 ** 2)  ) )

        b = mf1[index - 1] + mf2[index - 1] + mf3[index - 1]
        a1 = mf1[index - 1] * ( p1 * i + q1 )
        a2 = mf2[index - 1] * ( p2 * i + q2 )
        a3 = mf3[index - 1] * ( p3 * i + q3 )

        a = a1 + a2 +a3
        index += 1
        if b == 0:
            y.append(0)
        else:
            y.append(a / b)
    return y

def ga_fun(p):
    a, b, c, d, e, f, g, h, i, j, k, m = p
    residuals = np.float64(abs(my_function_ga(x_curve_fitting_real, a, b, c, d, e, f, g, h, i, j, k, m) - y_curve_fitting_real)).sum()
    return residuals

def ga_fun_temp(p):
    a, b, c, d, e, f, g, h, i, j, k, m = p
    residuals = np.float64(abs(my_function_ga(x_curve_fitting_real, a, b, c, d, e, f, g, h, i, j, k, m) - y_temperature_real)).sum()
    return residuals

generations = 100
ga = GA(func=ga_fun, n_dim=12, size_pop=600, max_iter=generations, prob_mut=0.1,
        lb=0, ub=20)

ga_temperature = GA(func=ga_fun_temp, n_dim=12, size_pop=600, max_iter=generations, prob_mut=0.1,
        lb=-4, ub=23)

best_params, residuals = ga.run()
best_params_temperature, residuals_temperature = ga_temperature.run()

# print('best_x:', best_params, '\n', 'best_y:', residuals)

y_predict_best_params = my_function_ga(x_curve_fitting_real, *best_params)
y_predict_best_params_temperature = my_function_ga(x_temperature_real, *best_params_temperature)
#print('y2', y_predict_best_params)
# print('-'*20)

error_ax = fig.gca()
error_ax.set_ylabel('')
error_ax.set_xlabel('Generation')
best_y_generation = ga.generation_best_Y
best_x_generation = ga.generation_best_X


best_y_generation_temp = ga_temperature.generation_best_Y
best_x_generation_temp = ga_temperature.generation_best_X


best_test = ga_temperature.all_history_Y
x_animation = np.linspace(1, generations, generations)

y_ticks = np.linspace(5, 50, 10)
for i in range(1, generations+1):
    precipitacion_ax1.grid()
    temperatura_ax2.grid()
    error_ax.grid()
    y_prediction_test = my_function_ga(x_curve_fitting_real, *best_x_generation[i - 1])
    precipitacion_ax1.plot(x_curve_fitting_real, y_curve_fitting_real, color='blue', linestyle='dashed', label='Precipitacion', linewidth=1)
    precipitacion_ax1.plot(x_curve_fitting_real, y_prediction_test, '-', color='green', label='Precipitacion - GA')


    y_prediction_test_temp = my_function_ga(x_temperature_real, *best_x_generation_temp[i - 1])
    temperatura_ax2.plot(x_temperature_real, y_temperature_real, color='red', linestyle='dashed', label='Temperatura', linewidth=1)
    temperatura_ax2.plot(x_temperature_real, y_prediction_test_temp, '-', color='orange', label='Temperatura - GA')


    temperatura_ax2.legend(loc="upper right")
    temperatura_ax2.set_yticks(y_ticks)
    precipitacion_ax1.legend(loc="upper right")
    precipitacion_ax1.set_yticks(y_ticks)


    error_ax.set_xlabel('Generation')

    data_animation_x = x_animation[:i]
    data_animation_y = best_y_generation[:i]
    data_animation_y_tem = best_y_generation_temp[:i]

    error_ax.plot(data_animation_x, data_animation_y, '-', color='green', label='Precipitacion - GA')
    error_ax.plot(data_animation_x, data_animation_y_tem, '-', color='orange', label='Temperatura - GA')
    error_ax.legend(loc="upper right")
    plt.pause(0.01)
    precipitacion_ax1.cla()
    temperatura_ax2.cla()
    error_ax.cla()


y_predict_best_params_converted = np.array(y_predict_best_params)
y_predict_best_params_converted_temp = np.array(y_predict_best_params_temperature)
precipitacion_ax1.grid()
precipitacion_ax1.plot(x_curve_fitting_real, y_curve_fitting_real, color='blue', linestyle='dashed', label='Precipitacion', linewidth=1)
precipitacion_ax1.plot(x_curve_fitting_real, y_predict_best_params_converted, '-', color='green', label='Precipitacion - GA')


temperatura_ax2.grid()
temperatura_ax2.plot(x_temperature_real, y_temperature_real, color='red', linestyle='dashed', label='Temperatura', linewidth=1)
temperatura_ax2.plot(x_temperature_real, y_predict_best_params_converted_temp, '-', color='orange', label='Temperatura - GA')

temperatura_ax2.legend(loc="upper right")
temperatura_ax2.set_yticks(y_ticks)
precipitacion_ax1.legend(loc="upper right")
precipitacion_ax1.set_yticks(y_ticks)

error_ax.grid()
error_ax.set_xlabel('Generation')
error_ax.plot(data_animation_x, data_animation_y, '-', color='green', label='Precipitacion - GA')
error_ax.plot(data_animation_x, data_animation_y_tem, '-', color='orange', label='Temperatura - GA')
error_ax.legend(loc="upper right")
plt.pause(0.01)
precipitacion_ax1.cla()
temperatura_ax2.cla()
error_ax.cla()
plt.ioff()

plt.show()