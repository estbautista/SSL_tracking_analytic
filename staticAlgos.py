import numpy as np
from numpy import inf
import time
import random

def GS_static(graph, eps, alpha, seed, method ) :
	
	# initial distribution
	N = graph.A.shape[0]
	y = np.zeros([N,1]); y[seed,0] = 1;

	# extract operator and coefficients
	if method == 'PageRank':
		rho = (1-alpha)
		psi = alpha;
		OP = graph.P.T
		gt = graph.pr.T
		deg = np.count_nonzero(OP, axis = 1)

	elif method == 'GammaPageRank':
		mu = (1-alpha)/alpha
		psi = -10/(2*mu + 10)
		rho = (2*mu)/(2*mu + 10)
		OP = graph.Op_shift
		gt = graph.gpr
		deg = np.count_nonzero(OP, axis = 1)
	
	# compute initial distribution (assuming p(0) = 0)
	p = np.zeros([N,1]);
	r = rho*y 
	it = 0;
	flops = 0

	#while np.linalg.norm(r, inf) > eps :
	while (np.linalg.norm(p - gt, ord=2)/np.linalg.norm(gt,ord=2))  > eps :
		it += 1;	
		ix_u = np.argmax(abs(r))
		r_u = float(r[ix_u]); 
		p[ix_u] += r_u
		r[ix_u] = 0; 
		r = r + np.expand_dims(psi*r_u*OP[:,ix_u],1)

		# count the flops 
		flops_scaling = np.count_nonzero(OP[:,ix_u])
		flops = flops + 2*flops_scaling + np.int(deg[ix_u])
	
	print('---- Gauss Soutwell ----')
	print('iter =', it)
	print('flops =', flops)
	print('err =', np.linalg.norm(p - gt,ord=2)/np.linalg.norm(gt,ord=2) )
	return p, r, it, flops

def PI_static(graph, eps, alpha, seed, method) :

	# initial distribution
	N = graph.A.shape[0]
	y = np.zeros([N,1]); y[seed,0] = 1;

	# extract operator and coefficients
	if method == 'PageRank':
		rho = (1-alpha)
		psi = alpha;
		OP = graph.P.T
		gt = graph.pr.T

	elif method == 'GammaPageRank':
		mu = (1-alpha)/alpha
		psi = -10/(2*mu + 10)
		rho = (2*mu)/(2*mu + 10)
		OP = graph.Op_shift
		gt = graph.gpr
	
	# compute initial distribution 
	p = np.zeros([N,1]);
	it = 0;
	flops = 0
	
	#for it in range(K):
	while (np.linalg.norm(p - gt, 2)/np.linalg.norm(gt,ord=2)) > eps :		
		it += 1
	
		# count flops
		nnz = np.where(p > 0)[0]
		flops_dotprod = np.count_nonzero(OP[:,nnz])	
		flops_scaling = np.count_nonzero(p)	
		flops_addition = np.count_nonzero(OP.dot(p))
		flops = flops + flops_dotprod + flops_scaling + flops_addition

		# next iteration in approx
		p = rho*y + psi*OP.dot(p)

	print('---- Power Iteration ----')
	print('iter =', it)
	print('flops =', flops)
	print('err =', np.linalg.norm(p - gt, 2)/np.linalg.norm(gt,ord=2))

	return p, it, flops

def CP_static(graph, eps, alpha, seed, method, mode) :
	# initial distribution
	N = graph.A.shape[0];
	y = np.zeros([N,1]); y[seed,0] = 1
	flops = 0

	# Coefficient parameters
	mu = (1-alpha)/alpha
	theta = np.linspace(0,np.pi,50000+1)
	step = theta[1] - theta[0]

	# extract operator and coefficients
	if method == 'PageRank':	
		Lap = (graph.D - graph.A).dot(graph.Dinv)
		OP = Lap - np.eye(N)
		gt = graph.pr.T
		filt_arg = (np.cos(theta) + 1)
		filt = np.divide(mu, mu + filt_arg)

	elif method == 'GammaPageRank':
		OP = graph.Op_shift
		gt = graph.gpr	
		filt_arg = (10/2)*(np.cos(theta) + 1)
		filt = np.divide(mu, mu + filt_arg)


	# coefficients
	tmp1 = np.multiply(np.cos(0*theta),filt*step); tmp1[0]=tmp1[0]/2; tmp1[-1]=tmp1[-1]/2;
	tmp2 = np.multiply(np.cos(1*theta),filt*step); tmp2[0]=tmp2[0]/2; tmp2[-1]=tmp2[-1]/2;
	coef1 = (2/np.pi)*np.sum(tmp1)
	coef2 = (2/np.pi)*np.sum(tmp2)

	# Polynomial elements
	polyTerm_2back = np.array(y)
	polyTerm_1back = np.array(OP).dot(y)	
	nnz = np.where(y != 0)[0]
	flops = flops + np.count_nonzero(OP[:,nnz])	

        # Chebyshev approximation
	Cheby_approximation_prev = 0.5*coef1*polyTerm_2back + coef2*polyTerm_1back;
	Cheby_approximation_curr = np.array(Cheby_approximation_prev)
	flops = flops + 2*np.count_nonzero(polyTerm_1back)

	#for it in range(2, hops-1):
	it = 2;
	activeNodes = np.where(graph.clust_memb > 0)[0]
	
	if mode == 'FixedError':
		while (np.linalg.norm(Cheby_approximation_curr[activeNodes] - gt[activeNodes], ord=2)/np.linalg.norm(gt[activeNodes],ord=2)) > eps :

			# Chebyshev coefficient
			tmp = np.array(np.multiply(np.cos(it*theta),filt*step)); tmp[0]=tmp[0]/2; tmp[-1]=tmp[-1]/2;
			coef_curr = (2/np.pi)*np.sum(tmp);

			# Current polynomial term
			polyTerm_curr = 2*(OP).dot(polyTerm_1back) - polyTerm_2back;
			nnz = np.where(polyTerm_1back != 0)[0]
			flops = flops + np.count_nonzero(OP[:,nnz]) + np.count_nonzero(OP.dot(polyTerm_1back))  + np.count_nonzero(polyTerm_1back) + np.count_nonzero(polyTerm_2back)

			# Chebyshev approximation
			Cheby_approximation_curr = np.array(Cheby_approximation_prev + coef_curr*polyTerm_curr);
			flops = flops + 2*np.count_nonzero(polyTerm_curr)

			# Update 
			polyTerm_2back = np.array(polyTerm_1back);
			polyTerm_1back = np.array(polyTerm_curr);
			Cheby_approximation_prev = np.array(Cheby_approximation_curr);
			it += 1
	elif mode == 'FixedFlops':
		while flops < eps:
			# Chebyshev coefficient
			tmp = np.array(np.multiply(np.cos(it*theta),filt*step)); tmp[0]=tmp[0]/2; tmp[-1]=tmp[-1]/2;
			coef_curr = (2/np.pi)*np.sum(tmp);

			# Current polynomial term
			polyTerm_curr = 2*(OP).dot(polyTerm_1back) - polyTerm_2back;
			nnz = np.where(polyTerm_1back != 0)[0]
			flops = flops + np.count_nonzero(OP[:,nnz]) + np.count_nonzero(OP.dot(polyTerm_1back))  + np.count_nonzero(polyTerm_1back) + np.count_nonzero(polyTerm_2back)

			# Chebyshev approximation
			Cheby_approximation_curr = np.array(Cheby_approximation_prev + coef_curr*polyTerm_curr);
			flops = flops + 2*np.count_nonzero(polyTerm_curr)

			# Update 
			polyTerm_2back = np.array(polyTerm_1back);
			polyTerm_1back = np.array(polyTerm_curr);
			Cheby_approximation_prev = np.array(Cheby_approximation_curr);
			it += 1

	p = Cheby_approximation_curr
	err = np.linalg.norm(p[activeNodes] - gt[activeNodes],ord=2)/np.linalg.norm(gt[activeNodes],ord=2)
	print('---- Chebyshev ----')
	print('iter =', it)
	print('flops =', flops)
	print('err =', err )

	return p, it, flops, err


