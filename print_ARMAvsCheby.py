import matplotlib.pyplot as plt
import scipy.io
mat = scipy.io.loadmat('Result_ARMAvsCheby.mat')

orders = (mat.get('orders')) 
err_arma = (mat.get('err_arma'))
err_cheby = (mat.get('err_cheby'))


plt.figure(figsize=[4,4])
plt.plot(orders[0], err_cheby[0], '-rs', label='Chebyshev')
plt.plot(orders[0], err_arma[0], '-g^', label='ARMA')
plt.xlabel('Approx. order (K)')
plt.ylabel('Relative error ($\ell_2$-norm)')
plt.yscale('log')
plt.legend()
plt.savefig('ARMAvsCheby_Warm', dpi=300, edgecolor='w',bbox_inches="tight") 
