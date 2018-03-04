import matplotlib.pyplot as plt
import numpy as np

a=[0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0]
b=np.array([0.0, 3.0, 2.5, 4.0, 6.0, 7.5, 7.0, 4.5, 12.0, 8.0, 13.0, 15.0, 10.0  ])

theta_0=0;
theta_1=1;

hyp=lambda x : theta_0 + theta_1*x                    #hypothesis equation

def plot_line(hyp, data_points):
    a_values = [i for i in range(int(min(data_points))-1, int(max(data_points))+2)]
    b_values = [hyp(x) for x in a_values]
    plt.plot(a_values, b_values, 'r')

Alpha=0.001                                           #Learning rate

def cost(hyp, a, b):                                  #cost function
    tot1=0
    tot2=0
    for i in range(1, len(a)):
        tot1 += hyp(a[i]) - b[i]
        tot2 += (hyp(a[i]) - b[i]) * a[i]
    return tot1 / len(a), tot2 / len(a)

for i in range(50):                                    #Gradient descet steps
    s1, s2 = cost(hyp, a, b)
    theta_1 = theta_1 - Alpha * s2
    theta_0 = theta_0 - Alpha * s1
    plot_line(hyp, a)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Dataset")
    plt.scatter(a, b, c='b')
    line_up, = plt.plot([1],marker='o', label='Data Ponits', c='b')
    line_down, = plt.plot([1], label='Pridiction', c='r')
    plt.legend(handles=[line_up, line_down])
plt.show()

print("Theta Values ")                                              #final theta values
print("theta0: ", theta_0, "theta1:", theta_1)

prid = hyp(b)                                                       #predict y and calculate mean squared error
MSE = 1/len(a) * sum((prid - b)**2)
print("MSE: ")
print(MSE)