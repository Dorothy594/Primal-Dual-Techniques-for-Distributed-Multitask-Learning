import numpy as np
import matplotlib.pyplot as plt

labels = ['alg1,n1', 'alg1,n2', 'alg1,mocha', 'FedAvg', 'lin_reg', 'd_tree', 'saga_ci']
x_pos = np.arange(len(labels))

mean_MSEs = [6.698781618047959e-06, 6.958639025609133e-06, 0.0399669506494894, 4.090825418709769, 4.044966941385034, 4.342326921745786, 0.008880541662046507]
std_MSEs = [6.910968912019921e-07, 5.418992988796196e-07, 0.0036041186560699053, 0.12075245594910722, 0.10960555817192678, 0.04580839153992601, 0.06355060362531235]

fig, ax = plt.subplots()
ax.bar(x_pos, mean_MSEs,
       yerr=std_MSEs,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=20)
ax.set_ylabel('MSE')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_yscale('log')
ax.set_title('error bars plot')
plt.show()
