from graph_data import graph_sequence
from staticAlgos import GS_static, PI_static, CP_static
from updateAlgos import GS_update, PI_update
import numpy as np
import matplotlib.pyplot as plt

#####################
## Graph parameters 
num_time_steps = 3
num_clust = 2
nodes_clust = 50

p_intra = 3/nodes_clust
p_inter = 0.1/nodes_clust

p_njoin = 0
p_nleave = 0
p_ejoin_in = 0.3*p_intra/nodes_clust
p_ejoin_out = 0.3*p_inter/(nodes_clust*(num_clust - 1))
p_eleave_in = 0.3*p_intra/nodes_clust
p_eleave_out = 0.3*p_inter/(nodes_clust*(num_clust - 1))
p_switch = 0

#####################
## Parameters GSSL
hops = 5;
snaps = 0;
mu = 1;
alpha = 1/(1+mu)
seed = range(0,10)
method = 'GammaPageRank'
mode = 'FixedError'

####################
## Store result
eps = 1e-3
flops_cps = np.zeros([num_time_steps-1,1])
flops_piu = np.zeros([num_time_steps-1,1])
flops_gsu = np.zeros([num_time_steps-1,1])
err_noup = np.zeros([num_time_steps,1])
change = np.zeros([num_time_steps,1])

####################
## Sequence of graphs
seq = graph_sequence(num_time_steps, num_clust, nodes_clust, p_njoin, p_nleave, p_ejoin_in, p_ejoin_out, p_eleave_in, p_eleave_out, p_switch, p_intra, p_inter, seed, alpha)

N = seq[0].A.shape[0]

# set the initial solution as the true solution
if method == 'PageRank':
	seq[0].p_gs = np.array(seq[0].pr.T)
	seq[0].r_gs = np.zeros([N,1])
	seq[0].p_pi = np.array(seq[0].pr.T)
elif method == 'GammaPageRank':
	seq[0].p_gs = np.array(seq[0].gpr)
	seq[0].r_gs = np.zeros([N,1])
	seq[0].p_pi = np.array(seq[0].gpr)

for ii in range(1,num_time_steps):	
	p_cps, it_cps, flops_cps[ii-1], err = CP_static(seq[ii], eps, alpha, seed, method, mode)
	seq[ii].p_pi, it_piu, flops_piu[ii-1], ch, err = PI_update(seq[ii-1], seq[ii], eps, alpha, method, 'track', mode)
	seq[ii].p_gs, seq[ii].r_gs, it_gsu, flops_gsu[ii-1], change[ii], err = GS_update(seq[ii-1], seq[ii], eps, alpha, method, 'track', mode)

print('flops Chebyshev', np.sum(flops_cps))
print('flops Power IT', np.sum(flops_piu))
print('flops Gauss SW', np.sum(flops_gsu))

print(flops_cps)
plt.plot(range(1,num_time_steps), np.cumsum(flops_cps) )
plt.plot(range(1,num_time_steps), np.cumsum(flops_piu) )
plt.plot(range(1,num_time_steps), np.cumsum(flops_gsu) )
plt.yscale('log')
plt.show()

