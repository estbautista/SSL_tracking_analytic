import numpy as np
from numpy import inf
import random

def GS_update(graph_init, graph_evol, eps, alpha, method, opt, mode) :
	
	# initial distribution
	N = graph_init.A.shape[0]	
	flops = 0
	
	# extract operator and coefficients
	if method == 'PageRank':
		mu = (1-alpha)/alpha
		psi = -2/(2*mu + 2)
		rho = (2*mu)/(2*mu + 2)
		OP_dif = np.array(-graph_evol.P.T + graph_init.P.T)
		OP_evol = np.array(-graph_evol.P.T)
		OP_init = np.array(-graph_init.P.T)
		gt_init = np.array(graph_init.pr.T)
		gt_evol = np.array(graph_evol.pr.T)
		deg = np.count_nonzero(graph_init.P.T, axis=1)
		
	elif method == 'GammaPageRank':
		mu = (1-alpha)/alpha
		psi = -10/(2*mu + 10)
		rho = (2*mu)/(2*mu + 10)
		OP_dif = np.array(graph_evol.Op_shift - graph_init.Op_shift)
		OP_evol = np.array(graph_evol.Op_shift)
		OP_init = np.array(graph_init.Op_shift)
		gt_init = np.array(graph_init.gpr)
		gt_evol = np.array(graph_evol.gpr)
		deg = np.count_nonzero(graph_evol.Op_shift, axis=1)

	print('---- Edges changed  ----')
	print('Edges init : ', np.count_nonzero(OP_dif - OP_evol) )
	print('Edges Changed : ', np.count_nonzero(OP_dif) )
	print('Edges evol : ', np.count_nonzero(OP_evol) )
	
	# compute initial distribution (assuming p(0) = 0)
	if opt == 'track' : 
		p = np.array(graph_init.p_gs)
		r = np.array(graph_init.r_gs + psi*(OP_dif.dot(p)))
		nnz = np.where(p != 0)[0]
		flops = flops + np.count_nonzero(p) + np.count_nonzero(OP_dif[:,nnz]) + np.count_nonzero(OP_dif.dot(p))

	elif opt == 'exact' :
		p = np.array(gt_init)
		r = np.array(psi*OP_dif.dot(p))
		nnz = np.where(p != 0)[0]
		flops = flops + np.count_nonzero(p) + np.count_nonzero(OP_dif[:,nnz]) + np.count_nonzero(OP_dif.dot(p))

	it = 0
	activeNodes = np.where(graph_evol.clust_memb > 0)[0]
	if mode == 'FixedError' :
		while (np.linalg.norm(p[activeNodes] - gt_evol[activeNodes] ,ord=2)/np.linalg.norm(gt_evol[activeNodes],ord=2)) > eps :
			if np.count_nonzero(OP_dif) == 0:
				break
			it += 1
			ix_u = np.argmax(abs(r))
			r_u = float(r[ix_u])
			p[ix_u] += r_u
			r[ix_u] = 0;
			r = r + np.expand_dims(psi*r_u*OP_evol[:,ix_u],1)
			flops = flops + 2*np.count_nonzero(OP_evol[:,ix_u]) + np.int(deg[ix_u])
		
	elif mode == 'FixedFlops':
		while flops < eps :

			if np.count_nonzero(OP_dif) == 0:
				break
			it += 1
			ix_u = np.argmax(abs(r))
			r_u = float(r[ix_u])
			p[ix_u] += r_u
			r[ix_u] = 0;
			r = r + np.expand_dims(psi*r_u*OP_evol[:,ix_u],1)
			flops = flops + 2*np.count_nonzero(OP_evol[:,ix_u]) + np.int(deg[ix_u])
	
	if it == 0:
		flops = 1

	change = np.count_nonzero(OP_dif)/np.count_nonzero(OP_init)
	err = np.linalg.norm(p[activeNodes] - gt_evol[activeNodes], ord=2)/np.linalg.norm(gt_evol[activeNodes])
	print('---- Gauss Soutwell Update ----')
	print('iter =', it)
	print('flops =', flops)
	print('err =', err)
	return p, r, it, flops, change, err

def PI_update(graph_init, graph_evol, eps, alpha, method, opt, mode):

	# initial distribution
	N = graph_init.A.shape[0]	
	flops = 0
	
	# extract operator and coefficients
	if method == 'PageRank':
		mu = (1-alpha)/alpha
		psi = -2/(2*mu + 2)
		rho = (2*mu)/(2*mu + 2)
		OP_dif = np.array( -graph_evol.P.T + graph_init.P.T )
		OP_evol = np.array( -graph_evol.P.T )
		OP_init = np.array( -graph_init.P.T )
		gt_init = np.array( graph_init.pr.T )
		gt_evol = np.array( graph_evol.pr.T )
		
	elif method == 'GammaPageRank':
		mu = (1-alpha)/alpha
		psi = -10/(2*mu + 10)
		rho = (2*mu)/(2*mu + 10)
		OP_dif = np.array( graph_evol.Op_shift - graph_init.Op_shift )
		OP_evol = np.array( graph_evol.Op_shift )
		OP_init = np.array( graph_init.Op_shift )
		gt_init = np.array( graph_init.gpr )
		gt_evol = np.array( graph_evol.gpr )

	print('---- Edges changed  ----')
	print('Edges init : ', np.count_nonzero(OP_dif - OP_evol) )
	print('Edges Changed : ', np.count_nonzero(OP_dif) )
	print('Edges evol : ', np.count_nonzero(OP_evol) )

	activeNodes = np.where(graph_evol.clust_memb > 0)[0]
	# compute initial distribution (assuming p(0) = 0)
	if opt == 'track' : 
		p = np.array( graph_init.p_pi )
		r = np.array( psi*(OP_dif.dot(p)) )
		nnz = np.where(p[activeNodes] != 0)[0]
		flops = flops + np.count_nonzero(OP_dif.dot(p)) + np.count_nonzero(OP_dif[:,nnz]) 

	elif opt == 'exact' :
		p = np.array( gt_init )
		r = np.array( psi*(OP_dif.dot(p)) )
		nnz = np.where(p[activeNodes] != 0)[0]
		flops = flops + np.count_nonzero(OP_dif.dot(p)) + np.count_nonzero(OP_dif[:,nnz]) 

	it = 0
	p_tmp = np.zeros([N,1])
	
	if mode == 'FixedError':
		while (np.linalg.norm(p[activeNodes] + (p_tmp[activeNodes]/rho) - gt_evol[activeNodes],ord=2)/np.linalg.norm(gt_evol[activeNodes],ord=2)) > eps :
			if np.count_nonzero(OP_dif) == 0:	
				break

			it += 1
			if it > 1e4:
				flops = float('NaN')
				p_tmp = np.zeros([N,1])
				break

			# count the flops
			nnz = np.where(p_tmp != 0)[0]
			flops_dotprod = np.count_nonzero(OP_evol[:,nnz])	
			flops_scaling = np.count_nonzero(p_tmp)	+ np.count_nonzero(r)
			flops_addition = np.count_nonzero(OP_evol.dot(p)) + np.count_nonzero(r)
			flops = flops + flops_dotprod + flops_scaling + flops_addition
		
			# next iteration
			p_tmp = rho*r + psi*OP_evol.dot(p_tmp)

	elif mode == 'FixedFlops':
		while flops < eps :
			if np.count_nonzero(OP_dif) == 0:	
				break

			it += 1

			# count the flops
			nnz = np.where(p_tmp != 0)[0]
			flops_dotprod = np.count_nonzero(OP_evol[:,nnz])	
			flops_scaling = np.count_nonzero(p_tmp)	+ np.count_nonzero(r)
			flops_addition = np.count_nonzero(OP_evol.dot(p)) + np.count_nonzero(r)
			flops = flops + flops_dotprod + flops_scaling + flops_addition
		
			# next iteration
			p_tmp = rho*r + psi*OP_evol.dot(p_tmp)

	p = p + (p_tmp/rho)
	flops = flops + np.count_nonzero(p_tmp)

	if it == 0:
		flops = 1

	change = np.count_nonzero(OP_dif)/np.count_nonzero(OP_init)
	err = np.linalg.norm(p[activeNodes] - gt_evol[activeNodes],ord=2)/np.linalg.norm(gt_evol[activeNodes],ord=2)
	print('---- Power Iteration Update ----')
	print('iter =', it)
	print('flops =', flops)
	print('err =', err)

	return p, it, flops, change, err
