import numpy as np 
import matplotlib.pyplot as plt
import kittiwrapper,util
import pptk
from tqdm import tqdm_notebook as tqdm
from neupy import algorithms
import open3d as o3
import arrow,scipy
import copy
import os
import matplotlib.cm as cm
import pyquaternion as pq
import pptk as pt
from scipy.signal import savgol_filter
from sklearn.mixture import GaussianMixture
#import C_Modifiedparticlefilter
import particlefilter
import progressbar
from sklearn.metrics import silhouette_score
import statistics
from sklearn.cluster import KMeans,DBSCAN,SpectralClustering,AgglomerativeClustering
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans as KM 
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from pso.pso import ParticleSwarmOptimizedClustering
from geographiclib.geodesic import Geodesic
from pandaset import DataSet
import utm


def get_bearing(lat1,long1,lat2, long2):
    brng = Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)['azi1']
    return brng

def handling_data(file,seq,date,drive):
    dataset = kittiwrapper.kittiwrapper(file)
    sequence = dataset.sequence(seq,date,drive)
    T_imu_cam0 = util.invert_ht(sequence.calib.T_cam0_imu)
    sequence.poses = np.array(
                    [np.matmul(oxts.T_w_imu, T_imu_cam0) \
                        for oxts in sequence.oxts])      #W-cam0
    T_w_velo =np.array([np.matmul(
                    sequence.poses[i,:,:], sequence.calib.T_cam0_velo) for i in range(len(sequence.poses))])
    T_w_velo_gt = np.matmul(sequence.poses, sequence.calib.T_cam0_velo)
    T_w_velo_gt = np.array([util.project_xy(ht) for ht in T_w_velo_gt])
    return sequence ,T_w_velo, T_w_velo_gt

def read_from_pandaset(path_to_data,seq_idx):
    dataset = DataSet(path_to_data)
    seq = dataset[seq_idx]
    pc_all = seq.load_lidar()
    lidar = pc_all.lidar
    poses = lidar.poses
    gps = seq.load_gps()
    gps = gps.gps
    n = len(gps[:])
    T_w_velo_gt=[]
    T_w_velo=[]
    angle1 = []
    #k=1
    for ele in range(0,n):
        try:
            angle = -np.radians(get_bearing(gps[ele-1]['lat'], gps[ele-1]['long'], gps[ele]['lat'], gps[ele]['long']))
        except:
            angle = 0.0
        x, y, zone, ut = utm.from_latlon(gps[ele]['lat'],gps[ele]['long'])
        angle1.append(angle)
        T_w_velo_gt.append(util.xyp2ht(np.array([x,y,angle])))
        a,b = poses[ele]['position']['x'],poses[ele]['position']['y']
        w = poses[ele]['heading']['w']
        T_w_velo.append(util.xyp2ht(np.array([a,b,w])))
    c = np.array(angle1)
    c[0]=0
    diffr = np.concatenate((np.array([[0,0]]),np.diff(np.array(T_w_velo_gt)[:, :3, 3], axis=0)[:,:2]))
    poses = np.cumsum(diffr,axis=0) 
    T_w_velo_gt1 = util.xyp2ht(np.hstack((poses,c.reshape(c.shape[0],1))))
                        
    return lidar, T_w_velo_gt1, np.array(T_w_velo)
    
def clustring_methods(numbre_of_clusters,X,model):
    if(model == 'fkm'):
        #range_n_clusters = [10, 20, 30, 40, 50, 60, 70, 80,90,100]
        #silhouette_avg = []
        #for num_clusters in range_n_clusters:

        #     # initialise kmeans
        kmeans = KM(k=num_clusters)
        kmeans.fit(X)
        cluster_labels = kmeans.labels_

             # silhouette score
        '''silhouette_avg.append(silhouette_score(X, cluster_labels))
        plt.plot(range_n_clusters,silhouette_avg,'bx-')
        plt.xlabel('number of clusters') 
        plt.ylabel('Silhouette score') 
        plt.title('Silhouette analysis For Optimal number of clusters')
        plt.savefig("0001.png")
        plt.show()
        print(arudb)'''
        #km = KM(k=numbre_of_clusters).fit(X)
        #return km.cluster_centers_


def features_extraction(n,sequence,T_w_velo,nodes_number,model, dataset_type):
    t = []
    for i in tqdm(range(n)):
        if(dataset_type == 0):
            datatest1 = sequence.get_velo(i)
            datatest = datatest1[:,:3]
            datatest = datatest [datatest[:,2]<0.2]
            datatest = datatest [datatest[:,2]>0]
            '''count, division = np.histogram(datatest[:,2],bins=10000, density=True)
            index = np.where(count==max(count))[0][0]
            a=division[index]
            extracting_relevent_data= []
            for j,pav in enumerate(datatest):
                if(a+1<datatest[j,2]):
                    extracting_relevent_data.append(pav)'''
            extracting_relevent_data = datatest
        else:
            datatest1 = sequence[i]
            datatest = np.vstack((np.array(datatest1['x']),np.array(datatest1['y']),np.array(datatest1['z']))).T
            datatest = datatest [datatest[:,2]<0.1]
            datatest = datatest [datatest[:,2]>0]
            extracting_relevent_data = datatest
        gngdata = clustring_methods(nodes_number,extracting_relevent_data,model)
        scan = o3.geometry.PointCloud()
        scan.points = o3.utility.Vector3dVector(gngdata)
        #cl, ind = scan.remove_statistical_outlier(nb_neighbors=2,std_ratio=1.0)
        #scan = scan.select_by_index(ind, invert = True)
        scan = scan.transform(T_w_velo[i])
        gngdata = np.array(scan.points)
        #plt.scatter(gngdata[:,0],gngdata[:,1],0.5) 
        t.append(gngdata)   
    return np.array(t)   

def create_ground_tru(n,t):
    scan = []
    for i in tqdm(range(n)):
        scan = np.concatenate((scan,t[i].reshape(t[i].shape[0]*t[i].shape[1])))
    groundtr = scan.reshape(int(scan.shape[0]/3),3) 
    # gm = GaussianMixture(n_components=1000,covariance_type='spherical', random_state=42).fit(groundtr)
    return groundtr

def view_global_map(features,T_w_velo_gt):
    plt.ion()
    figure = plt.figure()
    mapaxes = figure.add_subplot(1, 1, 1)
    mapaxes.scatter(features[:, 0], features[:, 1], s=5, c='b', marker='s')
    mapaxes.plot(T_w_velo_gt[:, 0, 3], T_w_velo_gt[:, 1, 3], 'g')
    particles = mapaxes.scatter([], [], s=1, c='r')
    arrow = mapaxes.arrow(0.0, 0.0, 3.0, 0.0, length_includes_head=True, 
        head_width=10, head_length=10, color='r')
    arrowdata = np.hstack(
        [arrow.get_xy(), np.zeros([8, 1]), np.ones([8, 1])]).T
    locpoles = mapaxes.scatter([], [], s=30, c='r', marker='o')
    viewoffset = 25.0
    plt.xlabel('x-coordinates in meter')
    plt.ylabel('y-coordinates in meter')
    plt.legend(["features points","vehicle's trajectory (m)"])
    plt.savefig('ex.png')

def Localization(n, groundtr,T_w_velo_gt,features,numparticle,range_of_pose,range_of_angle,estimate_type):
  
    polemap = groundtr
    i = 0
    polecov = 1.0
    filter = particlefilter.particlefilter(
             numparticle, T_w_velo_gt[i],range_of_pose, np.radians(range_of_angle), polemap, polecov)
    filter.minneff = 1
    filter.estimatetype = estimate_type	
    T_w_velo_est = np.full(T_w_velo_gt.shape, np.nan)
    T_w_velo_est[0] = filter.estimate_pose()
    for i in tqdm(range(1,n)):
        relodo = util.ht2xyp(
            util.invert_ht(T_w_velo_gt[i-1]).dot(T_w_velo_gt[i]))
        relodocov = np.diag((0.02 * relodo)**2)
        relodo = np.random.multivariate_normal(relodo, relodocov)
        filter.update_motion(relodo, relodocov)
        T_w_velo_est[i] = filter.estimate_pose()       
        poleparams = features[i]
        filter.update_measurement(poleparams[:, :2])
        T_w_velo_est[i] = filter.estimate_pose()
        filter.resample1()
    return T_w_velo_est   

def evaluation(n,T_w_velo_est,T_w_velo_gt):
    poserror = np.full([n, 1], np.nan)
    laterror = np.full([n, 1], np.nan)
    lonerror = np.full([n, 1], np.nan)
    angerror = np.full([n, 1], np.nan)
    T_gt_est = np.full([n, 4, 4], np.nan)
    for ieval in range(n):
        T_gt_est[ieval] = util.invert_ht(T_w_velo_gt[ieval]).dot(
            T_w_velo_est[ieval])
    lonerror[:, 0] = T_gt_est[:, 0, 3]
    laterror[:, 0] = T_gt_est[:, 1, 3]
    poserror[:, 0] = np.linalg.norm(T_gt_est[:, :2, 3], axis=1)
    angerror[:, 0] = util.ht2xyp(T_gt_est)[:, 2]
    angerror = np.degrees(angerror)
    lonstd = np.std(lonerror, axis=0)
    latstd = np.std(laterror, axis=0)
    angstd = np.std(angerror, axis=0)
    angerror = np.abs(angerror)
    laterror = np.mean(np.abs(laterror), axis=0)
    lonerror = np.mean(np.abs(lonerror), axis=0)
    posrmse = np.sqrt(np.mean(poserror ** 2, axis=0))
    angrmse = np.sqrt(np.mean(angerror ** 2, axis=0))
    poserror = np.mean(poserror, axis=0)
    angerror = np.mean(angerror, axis=0)
    print('poserror: {}\nposrmse: {}\n'
    'laterror: {}\nlatstd: {}\n'
    'lonerror: {}\nlonstd: {}\n'
    'angerror: {}\nangstd: {}\nangrmse: {}'.format(
        np.mean(poserror), np.mean(posrmse), 
        np.mean(laterror), np.mean(latstd), 
        np.mean(lonerror), np.mean(lonstd),
        np.mean(angerror), np.mean(angstd), np.mean(angrmse)))
    plt.plot(T_w_velo_est[:, 0, 3], T_w_velo_est[:, 1, 3])
    plt.plot(T_w_velo_gt[:,0,3],T_w_velo_gt[:,1,3])
