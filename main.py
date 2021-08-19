import matplotlib.pyplot as plt
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



# ax2.axis('off')

img = plt.imread("guadalajara-average-monthly-rain.png")
ax2.imshow(img, zorder=0, extent=[0, 1800, 0, 800] )

#plt.margins(0)
#plt.rcParams['axes.xmargin'] = 0
#plt.rcParams['axes.ymargin'] = 0
plt.show()


"""
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


# img = plt.imread("guadalajara-average-monthly-rain.png")
# fig, ax = plt.subplots()
# ax.imshow(img, extent=[0, 1200, 0, 800], )

fig = plt.figure()

fig.set_size_inches(18.5, 10.5)
#ax1 = axs[0]
#ax2 = axs[1]
#ax3 = axs[2]
#ax4 = axs[2]

#axs.plot(x, mf1)
#axs.plot(x, mf2)
#axs.plot(x,mf3)

ax1 = fig.add_subplot(221)


ax1.title.set_text('Temperatura')

print(y)
print(len(y))
ax1.plot( x, y )
ax1.grid()
#plt.margins(0)
#plt.rcParams['axes.xmargin'] = 0
#plt.rcParams['axes.ymargin'] = 0
plt.show()
"""


#08131185MX6