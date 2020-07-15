from graph_data import graph_sequence
from staticAlgos import GS_static, PI_static, CP_static
from updateAlgos import GS_update, PI_update
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

#####################
## Graph parameters 
num_time_steps = 1
num_clust = 2
nodes_clust = 250

p_intra = 3/nodes_clust
p_inter = 0.2/nodes_clust

p_njoin = 1
p_nleave = 1
p_ejoin_in = 0.01*p_intra/nodes_clust
p_ejoin_out = 0.01*p_inter/(nodes_clust*(num_clust - 1))
p_eleave_in = 0.01*p_intra/nodes_clust
p_eleave_out = 0.01*p_inter/(nodes_clust*(num_clust - 1))
p_switch = 0

#####################
## Parameters GSSL
hops = 5;
snaps = 0;
mu = 1;
alpha = 1/(1+mu)
seed = range(0,10)
realizations = 20
method = 'GammaPageRank'
mode = 'FixedError'

####################
## Store result
eps = [1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6]
flops_cps = np.zeros([realizations,len(eps)])
flops_piu = np.zeros([realizations,len(eps)])
flops_gsu = np.zeros([realizations,len(eps)])
err_noup = np.zeros([realizations,1])
change = np.zeros([realizations,1])

for ii in range(realizations):
	seq = graph_sequence(num_time_steps, num_clust, nodes_clust, p_njoin, p_nleave, p_ejoin_in, p_ejoin_out, p_eleave_in, p_eleave_out, p_switch, p_intra, p_inter, seed, alpha)

	if method == 'PageRank':
		err_noup[ii] = np.linalg.norm(seq[1].pr.T - seq[0].pr.T, ord=2)/np.linalg.norm(seq[1].pr.T, ord=2)
	elif method == 'GammaPageRank':
		err_noup[ii] = np.linalg.norm(seq[1].gpr - seq[0].gpr, ord=2)/np.linalg.norm(seq[1].gpr, ord=2)
	count = 0
	for epsilon in eps:	
		p_cps, it_cps, flops_cps[ii,count], err = CP_static(seq[1], epsilon, alpha, seed, method, mode)
		p_piu, it_piu, flops_piu[ii,count], ch, err = PI_update(seq[0], seq[1], epsilon, alpha, method, 'exact', mode)
		p_gsu, r_gsu, it_gsu, flops_gsu[ii,count], change[ii], err = GS_update(seq[0], seq[1], epsilon, alpha, method, 'exact', mode)
		count += 1

dictio = {'flops_cps': flops_cps  ,'flops_piu': flops_piu, 'flops_gsu': flops_gsu, 'eps': eps, 'change': change}
sio.savemat('Result_1evol.mat', dictio)

print('mean change', np.mean(change))

plt.figure(figsize=[4,4])
plt.loglog(eps,np.mean(flops_cps, axis=0), '-rs', label='Chebyshev')
plt.loglog(eps,np.mean(flops_piu, axis=0), '-g^', label='Update via PI')
plt.loglog(eps,np.mean(flops_gsu, axis=0), '-bo', label='Update via GS')
plt.xlabel('Approximation error')
plt.ylabel('Flops')
plt.legend()
plt.savefig('1evol', dpi=300, edgecolor='w',bbox_inches="tight") 
