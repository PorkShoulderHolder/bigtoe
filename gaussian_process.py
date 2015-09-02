import numpy as np
import cPickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn import gaussian_process


def load_data():
	x = np.load('../../../ibcdatagenerator_general/valid_lab_to_icd9_normalized.npy')
	print x.shape
	return x

def kernel(x,y,sigma=10,type=1):
	if type==1:
		#sigma is now the span on each side
		return max(0, 1-sigma*abs(x-y))

	if type==2:		
		return np.exp(-1.0*((x-y)**2)/(2*(sigma**2)) )

	if type==3:
		return np.exp(-1.0*(np.abs(x-y))/(2*(sigma**2)) )

	if type==4:
		#k = [0,0.1,0.2,0.5,0.6,0.3,0.2,0.1,0]
		k = [0.1,0.2,0.3,0.8,0.4,0.4,0.4]
		if int(np.abs(x-y)/sigma) < len(k):
			return k[int(np.abs(x-y)/sigma)]
		else:
			return 0

	print 'kernel not defined.'
	return -1



def gp_setup(x, type=1, sigma=10, gp_noise_sigma=0.1, personix=47, labix=3, inputx=[]):
	fig = plt.figure()
	ims = []
	
	if len(inputx) == 0:
		x = load_data()
		x1 = x[ labix, personix,:]
	else:
		#x = inputx
		x1 = inputx

	t1 = x1.nonzero()[0]
	x1 = x1[t1]
	mu = x1.mean()
	x1 = x1-mu
	std = x1.std()
	x1 = x1/std
	#print(t1,x1)	
	plt.scatter(t1, x1*std + mu)

	for sigma in np.array(range(1, 700))/10.0:
		# kernelMatrix1 = np.zeros((t1.shape[0], t1.shape[0]), dtype=float)
		# for i in range(0,kernelMatrix1.shape[0]):
		# 	for j in range(0, kernelMatrix1.shape[1]):
		# 		kernelMatrix1[i,j] = kernel(t1[i], t1[j], sigma, type)
		# #print kernelMatrix1
		# kernelMatrix1inv_smooth = np.linalg.inv( kernelMatrix1 + (gp_noise_sigma**2)*np.identity(kernelMatrix1.shape[0]) )
		# #print kernelMatrix1inv_smooth
		# kernelMatrix1inv_smooth_times_y = np.dot(kernelMatrix1inv_smooth, x1.transpose())
		# #print kernelMatrix1inv_smooth_times_y

		tnew = np.array(range(0,109))
		kernel_tnew = np.zeros((tnew.shape[0], t1.shape[0]), dtype=float)
		for i in range(0, tnew.shape[0]):
			for j in range(0, x1.shape[0]):
				kernel_tnew[i,j] = kernel(tnew[i], t1[j],  sigma, type) + kernel(tnew[i], t1[j], sigma, type)
		#print (kernel_tnew.sum(axis=1)*1.0)
		# xtnew_mean =  std*( np.dot(kernel_tnew, kernelMatrix1inv_smooth_times_y)) + mu
		#print xtnew_mean

		kernel_regressed_xnew = std*(np.dot(kernel_tnew, x1) / (kernel_tnew.sum(axis=1)*1.0)) + mu
		kernel_regressed_xnew[np.isnan(kernel_regressed_xnew)] = mu #was devided by zero it means no answer exists..
		#print kernel_regressed_xnew

		gp_nuggets = np.ones((t1.shape[0])) * (gp_noise_sigma**2)
		if type==1:		
			corr1 ='linear'
			theta00 = np.array([1.0/sigma])
		if type>=2:
			corr1 = 'squared_exponential'
			theta00 = [1.0/(2*(sigma**2))]
			
		# gp = gaussian_process.GaussianProcess(corr=corr1, theta0=theta00, nugget=gp_nuggets)
		# gp.fit([[t1[i]] for i in range(0,t1.shape[0])] , x1)
		# gp_xtnew = std*(gp.predict([[tnew[i]] for i in range(0,tnew.shape[0])])) + mu
		# #print gp_xtnew
		
		#im, =plt.plot(tnew, xtnew_mean, 'blue', label='my gp')
		im, = plt.plot(tnew, kernel_regressed_xnew, 'red', label='kernel regression sigma='+str(sigma))
		ims.append([im])
		#plt.plot(tnew, gp_xtnew, 'green', label=' scikit gp')
	#plt.legend(loc='upper right')
	ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
	#ani.save('regaussian_reg_increasing_sigma.mp4')	
	plt.show()
	return 0; #tnew, t1, x1+mu, xtnew_mean, kernel_regressed_xnew, gp_xtnew



def RMSEs_compare():
	x = load_data()
	x = x[:, 0:200, :]
	
	sigma_list = [10]#[0.05, 0.1, 0.5, 1, 5, 10, 25, 50, 100]
	gp_noise_sigma_list = [1] #[0.001, 0.01, 0.1, 1, 10]
	type_list = [2] #[1, 2] #,3
	#type=1, sigma=10, gp_noise_sigma=0.1
	gpscikit = np.zeros((x.shape[0], x.shape[1], x.shape[2], len(sigma_list), len(type_list), len(gp_noise_sigma_list)) , dtype=float)
	gpnarges = np.zeros((x.shape[0], x.shape[1], x.shape[2], len(sigma_list), len(type_list), len(gp_noise_sigma_list)) , dtype=float)
	kernelregression = np.zeros((x.shape[0], x.shape[1], x.shape[2], len(sigma_list), len(type_list)), dtype=float)

	rmse_scikit = np.zeros((x.shape[0], len(sigma_list), len(type_list), len(gp_noise_sigma_list)), dtype=float)
	rmse_gpnarges = np.zeros((x.shape[0], len(sigma_list), len(type_list), len(gp_noise_sigma_list)), dtype=float)
	rmse_kregression = np.zeros((x.shape[0], len(sigma_list), len(type_list)), dtype=float)
	counter_gp = np.zeros((x.shape[0], len(sigma_list), len(type_list), len(gp_noise_sigma_list)), dtype=float)
	counter_reg = np.zeros((x.shape[0], len(sigma_list), len(type_list)), dtype=float)

	print 'starting imputation..'
	for personix in range(0,x.shape[1]):
		print personix
		for labix in range(0, x.shape[0]):
			xx = x[labix, personix, :]
			tt = xx.nonzero()[0]
			xx = xx[tt]			
			if tt.shape[0]<3:
				continue

			for timeix in range(0,tt.shape[0]):								
				tmissing = tt[timeix]
				xtmissing = xx[timeix]
				x1 = np.delete(xx, timeix)
				t1 = np.delete(tt, timeix)
				mu = x1.mean()
				x1 = x1-mu
				std = x1.std()
				if std != 0: 
					x1 = x1/std
				#print '-x-',xtmissing

				for typeix, type in enumerate(type_list):
					for sigmaix, sigma in enumerate(sigma_list):					
						kernelMatrix1 = np.zeros((t1.shape[0], t1.shape[0]), dtype=float)
						for i in range(0,kernelMatrix1.shape[0]):
							for j in range(0, kernelMatrix1.shape[1]):
								kernelMatrix1[i,j] = kernel(t1[i], t1[j], sigma, type)

						tnew = np.array([tmissing])
						kernel_tnew = np.zeros((tnew.shape[0], t1.shape[0]), dtype=float)
						for i in range(0, tnew.shape[0]):
							for j in range(0, kernelMatrix1.shape[0]):
								kernel_tnew[i,j] = kernel(tnew[i], t1[j],  sigma, type)

						#first kernel regression
						kernel_regressed_xnew = std*(np.dot(kernel_tnew, x1) / (kernel_tnew.sum(axis=1)*1.0)) + mu
						kernel_regressed_xnew[np.isnan(kernel_regressed_xnew)] = mu
						kernelregression[labix, personix, tmissing, sigmaix, typeix] = kernel_regressed_xnew.ravel()[0]
						rmse_kregression[labix,sigmaix, typeix] += (kernelregression[ labix, personix, tmissing, sigmaix, typeix] - xtmissing)**2
						counter_reg[labix, sigmaix, typeix] += 1
						#print kernelregression[ labix, personix, tmissing, sigmaix, typeix]

						for gp_noise_ix, gp_noise_sigma in enumerate(gp_noise_sigma_list):
							#second GP regression
							gp_nuggets = np.ones((t1.shape[0])) * (gp_noise_sigma)
							if type==1:		
								corr1 ='linear'
								theta00 = np.array(sigma)
							if type==2:
								corr1 = 'squared_exponential'
								theta00 = np.array(sigma)
							if type==3:
								corr1 = 'absolute_exponential'
								theta00 = np.array(sigma)

							gp = gaussian_process.GaussianProcess(corr=corr1, theta0=theta00, nugget=gp_nuggets)
							
							gp.fit([[t1[i]] for i in range(0, t1.shape[0])] , x1)
							gp_xtnew = std*(gp.predict([[tnew[i]] for i in range(0,tnew.shape[0])])) + mu
							gpscikit[ labix, personix, tmissing, sigmaix, typeix, gp_noise_ix] = gp_xtnew.ravel()[0]
							rmse_scikit[labix,sigmaix, typeix, gp_noise_ix] += (gpscikit[ labix, personix, tmissing, sigmaix, typeix, gp_noise_ix] - xtmissing)**2
							#print gpscikit[ labix, personix, tmissing, sigmaix, typeix, gp_noise_ix]

							#third my GP regression:
							kernelMatrix1inv_smooth = np.linalg.inv( kernelMatrix1 + (gp_noise_sigma)*np.identity(kernelMatrix1.shape[0]) )
							kernelMatrix1inv_smooth_times_y = np.dot(kernelMatrix1inv_smooth, x1.transpose())
							xtnew_mean = std*( np.dot(kernel_tnew, kernelMatrix1inv_smooth_times_y)) + mu
							gpnarges[labix, personix, tmissing, sigmaix, typeix, gp_noise_ix] = xtnew_mean.ravel()[0]
							rmse_gpnarges[labix, sigmaix, typeix, gp_noise_ix] += (gpnarges[ labix, personix, tmissing, sigmaix, typeix, gp_noise_ix] - xtmissing)**2
							counter_gp[labix, sigmaix, typeix, gp_noise_ix] += 1.0
							#print gpnarges[ labix, personix, tmissing, sigmaix, typeix, gp_noise_ix]
	return x, kernelregression, gpnarges, gpscikit, np.sqrt(rmse_kregression/counter_reg), np.sqrt(rmse_scikit/counter_gp), np.sqrt(rmse_gpnarges/counter_gp), sigma_list, gp_noise_sigma_list, type_list



def plot_imputations(x, kernelreg, gpsci, gpnarges):
	nnzt = x.nonzero()[0]
	x = x[nnzt]
	kernelreg = kernelreg[nnzt]
	gpsci = gpsci[nnzt]
	gpnarges = gpnarges[nnzt]
	plt.scatter(nnzt, x)
	plt.plot(nnzt, kernelreg, 'blue', label='kernel regression')
	plt.plot(nnzt, gpnarges, 'red', label='gp narges')
	plt.plot(nnzt, gpsci, 'green', label='gp scikit')
	plt.legend(loc='upper right')
	plt.show()


def analyse_rmses():
	x, kernelregression, gpnarges, gpscikit, rmsereg, rmsesci, rmsenar, sigma_list, gp_noise_sigma_list, type_list = RMSEs_compare()
	f = open('../../../../../baseline5mil/config/loinc_file.top1000.withLabels', 'rb')
	labels = f.readlines()
	labels = [l.strip('\n').split('#')[1] for l in labels]
	f.close()

	for labix in range(0,x.shape[0]):
		print labix, labels[labix]

		bestix = np.unravel_index(rmsereg[labix].argmin(), rmsereg[labix].shape)
		print 'kreg best:', bestix, '(sigma=',sigma_list[bestix[0]], ' type=',type_list[bestix[1]], ') rmse:',  rmsereg[labix,bestix[0], bestix[1]]

		bestix = np.unravel_index(rmsesci[labix].argmin(), rmsesci[labix].shape)
		print 'gp-scikit best:', bestix, '(sigma=',sigma_list[bestix[0]], ' type=',type_list[bestix[1]], ' diag_sigma=',gp_noise_sigma_list[bestix[2]],') rmse:',  rmsesci[labix,bestix[0], bestix[1], bestix[2]]

		bestix = np.unravel_index(rmsenar[labix].argmin(), rmsenar[labix].shape)
		print 'gp-narges best:', bestix, '(sigma=',sigma_list[bestix[0]], ' type=',type_list[bestix[1]], ' diag_sigma=',gp_noise_sigma_list[bestix[2]],') rmse:',  rmsenar[labix,bestix[0], bestix[1], bestix[2]]

	return x, kernelregression, gpnarges, gpscikit, rmsereg, rmsesci, rmsenar, sigma_list, gp_noise_sigma_list, type_list