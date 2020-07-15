from graph_data import graph_sequence
from staticAlgos import GS_static, PI_static, CP_static
from updateAlgos import GS_update, PI_update
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


#####################
## Graph parameters 
num_time_steps = 30
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
method = 'GammaPageRank'
mode = 'FixedFlops'

####################
## Store result
flops = 1e4
err_cps = np.zeros([num_time_steps-1,1])
err_piu = np.zeros([num_time_steps-1,1])
err_gsu = np.zeros([num_time_steps-1,1])
err_noup = np.zeros([num_time_steps,1])
change = np.zeros([num_time_steps,1])


####################
## Sequence of graphs
seq = graph_sequence(num_time_steps, num_clust, nodes_clust, p_njoin, p_nleave, p_ejoin_in, p_ejoin_out, p_eleave_in, p_eleave_out, p_switch, p_intra, p_inter, seed, alpha)

# Graph size 
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
	p_cps, it_cps, fl, err_cps[ii-1] = CP_static(seq[ii], flops, alpha, seed, method, mode)
	print('flops Chebyshev:', fl)
	seq[ii].p_pi, it_piu, fl, ch, err_piu[ii-1] = PI_update(seq[ii-1], seq[ii], flops, alpha, method, 'track', mode)
	print('flops Power IT:', fl)
	seq[ii].p_gs, seq[ii].r_gs, it_gsu, fl, change[ii], err_gsu[ii-1] = GS_update(seq[ii-1], seq[ii], flops, alpha, method, 'track', mode)
	print('flops Gauss St:', fl)

dictio = {'err_cps': err_cps  ,'err_piu': err_piu, 'err_gsu': err_gsu, 'flops': flops, 'change': change}
sio.savemat('Result_fixedFlops.mat', dictio)

print('mean change', np.mean(change))

plt.figure(figsize=[4,4])
plt.plot(range(1,num_time_steps), err_cps, '-rs', label='Chebyshev')
plt.plot(range(1,num_time_steps), err_piu, '-g^', label='Update via PI' )
plt.plot(range(1,num_time_steps), err_gsu, '-bo', label='Update via GS' )
plt.xlabel('Graph snapshot')
plt.ylabel('Approximation error')
plt.yscale('log')
plt.legend()
plt.savefig('fixedFlops', dpi=300, edgecolor='w',bbox_inches="tight") 

