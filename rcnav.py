from numpy import *
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import mplleaflet
import utm
import mpld3

pd.options.mode.chained_assignment = None

'''class channel(object):

	def __init__(self):
	        pass
	def set_interval(self.interval):
		self.interval = interval
	def set_reference(self,reference):
		self.reference = reference
	def set_convoy_length(self,convoy_length):
		self.convoy_length = convoy_length
	def set_stretch(self, stretch):
		self.stretch = stretch
	def set_smooth_parameter(self, smooth_parameter):
		self.smooth_parameter = smooth_parameter
	def set_clusters_eps(self,eps):
		self.clusters_eps = eps
	def set_clusters_min_samples(self, clusters_min_samples):
		self.clusters_min_samples = clusters_min_samples'''

class rcnav(object):
	def __init__(self,x,y):
		self.label       = "Calculate radius of curvature"
		self.description = "Calculates the radius of curvature of a discrete path (x,y) in metric coordinate system." \
		'''	rcnav.py creates from a set of X,Y ordered route coordinates a dataframe with the local radii of curvatures, clusters, and critical turns center points.
			rcnav..findRC() - create self.df (a dataframe) with the radius of curvature of each 3 point set.
			rcnav.cluster(eps,min_samples,limit) - it is based on sklearn.cluster.DBSCAN. Creates a self.mp dataframe with Validador and Validador2 that define the critical points and the center of each critical cluster.
				eps : float : The maximum distance between two samples for them to be considered as in the same neighborhood.
				min_samples : int : The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
				limit : float : The maximum radius of curvature considered critic (meters).'''
		self.X = x
		self.Y = y
		df = pd.DataFrame()
		df['X'] = x
		df['Y'] = y
		self.df = df
        pass
	def findRC(self):
		
		mp = pd.DataFrame()
		mp['Y'] = self.Y
		mp['X'] = self.X
		mp['RC'] = np.nan
		mp['Ds'] = 0.0
		mp['Ds'][1:len(mp)]= map(lambda x: np.linalg.norm(array([mp.X[x],mp.Y[x]]) - array([mp.X[x-1],mp.Y[x-1]])),range(1,len(mp)))
		mp['dys'] = np.nan
		mp['dxs'] = np.nan
		mp['d2ys'] = np.nan
		mp['d2xs'] = np.nan
		mp['deno'] = np.nan
		mp['RC2'] = np.nan
		mp['radiusx'] = np.nan
		mp['radiusy'] = np.nan
		mp['xc'] = np.nan
		mp['yc'] = np.nan
		mp['dyx'] = np.nan
		#CALCULATION
		mp['dys'][1:len(mp)-1] = map(lambda x: (mp.Y[x+1] - mp.Y[x-1])/(2.0*((mp.Ds[x]+mp.Ds[x+1])/2.0)),range(1,len(mp)-1))
		mp['dxs'][1:len(mp)-1] = map(lambda x: (mp.X[x+1] - mp.X[x-1])/(2.0*((mp.Ds[x]+mp.Ds[x+1])/2.0)),range(1,len(mp)-1))
		mp['d2ys'][1:len(mp)-1] = map(lambda i: (mp.Y[i+1] - 2.0 * mp.Y[i] + mp.Y[i-1])/((mp.Ds[i]+mp.Ds[i+1])/2.0)**2 ,range(1,len(mp)-1))
		mp['d2xs'][1:len(mp)-1] = map(lambda i: (mp.X[i+1] - 2.0 * mp.X[i] + mp.X[i-1])/((mp.Ds[i]+mp.Ds[i+1])/2.0)**2 ,range(1,len(mp)-1))
		mp['deno'][1:len(mp)-1] = map(lambda i: sqrt((mp.dxs[i]*mp.d2ys[i]-mp.dys[i]*mp.d2xs[i])**2) ,range(1,len(mp)-1))
		mp['RC2'][1:len(mp)-1] = map(lambda i:((mp.dxs[i]**2 + mp.dys[i]**2)**(3.0/2))/(mp.deno[i]),range(1,len(mp)-1))
		#VALIDATION
		mp['dyx'][1:len(mp)-1] = map(lambda i: mp.dys[i]*(1 / mp.dxs[i]),range(1,len(mp)-1))
		mp.RC2[mp.RC2==np.inf] = 1000000000.0
		mp.RC2[mp.RC2.isnull()] = 1000000000.0
		mp.dyx[mp.dyx.isnull()] = 0.00000000000000000001
		mp['coeficiente_a'] = 1+(1.0/mp.dyx)**2
		mp['coeficiente_b'] = -2*mp.X*(1+(1/(-mp.dyx))**2)
		mp['coeficiente_c'] = (1+(1.0/(-mp.dyx))**2)*(mp.X)**2 - mp.RC2**2
		mp['Coeff'] = np.nan
		mp['X_centro1'] = np.nan
		mp['Y_centro1'] = np.nan
		mp['X_centro2'] = np.nan
		mp['Y_centro2'] = np.nan
		mp['radiusx1'] = np.nan
		mp['radiusy1'] = np.nan
		mp['radiusx2'] = np.nan
		mp['radiusy2'] = np.nan
		mp['Coeff'] = map(lambda i: [mp['coeficiente_a'][i],mp['coeficiente_b'][i],mp['coeficiente_c'][i]], range(len(mp)))
		listaauxiliar0 = map(lambda i: np.roots(mp['Coeff'][i])[1], range(1,len(mp)-1)) 
		listaauxiliar1 = map(lambda i: np.roots(mp['Coeff'][i])[0], range(1,len(mp)-1))
		mp['X_centro1'][1:len(mp)-1] = listaauxiliar0
		mp['X_centro2'][1:len(mp)-1] = listaauxiliar1
		mp['Y_centro1'] =-1/mp.dyx *(mp.X_centro1- mp.X)+ mp.Y
		mp['Y_centro2'] =-1/mp.dyx *(mp.X_centro2- mp.X)+ mp.Y
		mp['radiusx1'] = map(lambda i: [mp['X'][i],mp['X_centro1'][i]], range(len(mp)))
		mp['radiusy1'] = map(lambda i: [mp['Y'][i],mp['Y_centro1'][i]], range(len(mp)))
		mp['radiusx2'] = map(lambda i: [mp['X'][i],mp['X_centro2'][i]], range(len(mp)))
		mp['radiusy2'] = map(lambda i: [mp['Y'][i],mp['Y_centro2'][i]], range(len(mp)))
		mp['D'] = 0.0
		mp['D'] = mp.Ds.cumsum()
		self.df = mp
	def cluster(self,eps,min_samples,limit):
		cp0 = self.df[self.df.RC2<limit]
		cp00 = np.array([cp0.D, np.zeros(len(cp0))]).transpose()
		mp = self.df
		####

		db = DBSCAN(eps=eps,min_samples =min_samples).fit(cp00)
		core_samples_mask = np.zeros_like (db.labels_, dtype=bool)
		core_samples_mask[db.core_sample_indices_] = True
		labels = db.labels_
		core_samples_mask[db.core_sample_indices_] = True
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		print('Estimated number of clusters: %d' % n_clusters_)
		unique_labels = set(labels)
		colors = plt.cm.Set3(np.linspace (0, 1, len(unique_labels)))
		f2 = plt.figure(figsize=(400,3))
		ax = f2.add_subplot(111)
		li = list()
		lii = list()
		for k, col in zip(unique_labels, colors):
		    if k == -1:
		        # Black used for noise.
		        col = 'k'
		    class_member_mask = (labels == k)
		    xy = cp00[class_member_mask & core_samples_mask]
		    li.append(xy)
		    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
		             markeredgecolor='none', markersize=7, label='Passo'+str(k))
		    xy = cp00[class_member_mask & ~core_samples_mask]
		    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
		             markeredgecolor='none', markersize=7)
		    lii.append(xy)
		ct = li
		ct = li
		len(li)
		for i in range(len(ct)):
		    if i ==0 :
		        RC = tuple(x[0] for x in ct[i])
		        P = tuple(x[1] for x in ct[i])
		        I = tuple(i for x in ct[i])
		        Aux = pd.Series(RC)
		        Aux2 = pd.Series(P)
		        Aux3 = pd.Series(I)
		        RCF = Aux 
		        PF = Aux2
		        IF = Aux3
		    else:
		        RC = tuple(x[0] for x in ct[i])
		        P = tuple(x[1] for x in ct[i])
		        I = tuple(i for x in ct[i])
		        Aux = pd.Series(RC)
		        Aux2 = pd.Series(P)
		        Aux3 = pd.Series(I)
		        RCF = pd.concat([RCF,Aux], axis=0, join='outer', join_axes=None, ignore_index=True,
		               keys=None, levels=None, names=None, verify_integrity=False)
		        PF = pd.concat([PF,Aux2], axis=0, join='outer', join_axes=None, ignore_index=True,
		               keys=None, levels=None, names=None, verify_integrity=False)
		        IF = pd.concat([IF,Aux3], axis=0, join='outer', join_axes=None, ignore_index=True,
		               keys=None, levels=None, names=None, verify_integrity=False)
		CLS4 = pd.DataFrame(data = [RCF,PF,IF], columns= ['RC','P','I'])
		CLS4 = pd.DataFrame(RCF)
		CLS4.columns = ['S']
		CLS4['RC'] = PF
		CLS4['I'] = IF
		mp['Validador'] = np.nan
		mp['Validador'][(mp['D'].astype(int)).isin((CLS4.S.astype(int)).tolist())] = 1
		mp['Grupo'] = np.nan
		mp['Grupo'][(mp['D'].astype(int)).isin((CLS4.S.astype(int)).tolist())] = CLS4.sort(['S'],ascending=1).I.tolist()
		mp['Validador2'] = 0
		for i in range(len(li)):
		    mp.Validador2[mp.index[mp.RC2 == mp.RC2[mp.Grupo==i].min()]] = 1
		self.mp = mp
	def map(self,utm_fuse,utm_zone,mappath):
		fig = plt.figure()
		ax = plt.gca()

		a2 = self.mp.X[self.mp.Validador == 1]
		b2 = self.mp.Y[self.mp.Validador == 1]

		a = self.mp.X[self.mp.Validador2 == 1]
		b = self.mp.Y[self.mp.Validador2 == 1]

		pc = pd.DataFrame(map(lambda x: utm.to_latlon(a.tolist()[x], b.tolist()[x], utm_fuse, utm_zone),range(len(a)) ))
		pc2 = pd.DataFrame(map(lambda x: utm.to_latlon(a2.tolist()[x], b2.tolist()[x], utm_fuse, utm_zone),range(len(a2)) ))
		pt = pd.DataFrame(map(lambda x: utm.to_latlon(self.mp.X[x], self.mp.Y[x], utm_fuse, utm_zone),range(len(self.mp)) ))

		ax.scatter(pt[1],pt[0],c='b')
		ax.scatter(pc2[1],pc2[0],c='y',s=30)

		critic = ax.scatter(pc[1],pc[0],c='r',s=30)
		labels = self.mp.RC2[self.mp.Validador2 == 1].astype(str).tolist()
		tooltip = mpld3.plugins.PointLabelTooltip(critic, labels=labels)

		mpld3.plugins.connect(fig, tooltip)
		mplleaflet.show(path=mappath)








