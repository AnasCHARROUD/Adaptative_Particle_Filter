#!/usr/bin/env python

import warnings

import numpy as np
import scipy

import util


class particlefilter:
    def __init__(self, count, start, posrange, angrange, 
            polemeans, polevar, T_w_o=np.identity(4)):
        self.p_min = 0.01
        self.d_max = np.sqrt(-2.0 * polevar * np.log(
            np.sqrt(2.0 * np.pi * polevar) * self.p_min))
        self.minneff = 0.5
        self.estimatetype = 'best'
        self.count = count
        ############### particle initialization 
        r = np.random.uniform(low=0.0, high=posrange, size=[self.count, 1])
        angle = np.random.uniform(low=-np.pi, high=np.pi, size=[self.count, 1])
        xy = np.sqrt(r) * np.hstack([np.cos(angle), np.sin(angle)])
        dxyp = np.hstack([xy, np.random.uniform(
            low=-angrange, high=angrange, size=[self.count, 1])])
        self.particles = np.matmul(start, util.xyp2ht(dxyp))
        self.weights = np.full(self.count, 1.0 / self.count)
        self.polemeans = polemeans
        self.poledist = scipy.stats.norm(loc=0.0, scale=np.sqrt(polevar))
        self.kdtree = scipy.spatial.cKDTree(polemeans[:, :2], leafsize=3)
        self.T_w_o = T_w_o
        self.T_o_w = util.invert_ht(self.T_w_o)
        self.posrange = posrange
        self.angrange = angrange

    @property
    def neff(self):
        return 1.0 / (np.sum(self.weights**2.0) * self.count)

    def update_motion(self, mean, cov):
        T_r0_r1 = util.xyp2ht(
            np.random.multivariate_normal(mean, cov, self.count))
        self.particles = np.matmul(self.particles, T_r0_r1)

    def update_measurement(self, poleparams, resample=True):
        n = poleparams.shape[0]
        polepos_r = np.hstack(
            [poleparams[:, :2], np.zeros([n, 1]), np.ones([n, 1])]).T
        for i in range(self.count):
            polepos_w = self.particles[i].dot(polepos_r)
            d, _ = self.kdtree.query(
                polepos_w[:2].T, k=1, distance_upper_bound=self.d_max)
            self.weights[i] *= np.prod(
                self.poledist.pdf(np.clip(d, 0.0, self.d_max)) + 0.1)
        self.weights /= np.sum(self.weights)
        
        if resample and self.neff < self.minneff:
            self.resample()

    def estimate_pose(self):
        if self.estimatetype == 'mean':
            xyp = util.ht2xyp(np.matmul(self.T_o_w, self.particles))
            mean = np.hstack(
                [np.average(xyp[:, :2], axis=0, weights=self.weights),
                    util.average_angles(xyp[:, 2], weights=self.weights)])
            return self.T_w_o.dot(util.xyp2ht(mean))
        if self.estimatetype == 'max':
            return self.particles[np.argmax(self.weights)]
        if self.estimatetype == 'best':
            i = np.argsort(self.weights)[-int(0.1 * self.count):]
            xyp = util.ht2xyp(np.matmul(self.T_o_w, self.particles[i]))
            mean = np.hstack(
                [np.average(xyp[:, :2], axis=0, weights=self.weights[i]),
                    util.average_angles(xyp[:, 2], weights=self.weights[i])])                
            return self.T_w_o.dot(util.xyp2ht(mean))

    def resample(self):
        cumsum = np.cumsum(self.weights)
        pos = np.random.rand() / self.count
        idx = np.empty(self.count, dtype=np.int)
        ics = 0
        for i in range(self.count):
            while cumsum[ics] < pos:
                ics += 1
            idx[i] = ics
            pos += 1.0 / self.count
        self.particles = self.particles[idx]
        self.weights[:] = 1.0 / self.count
        
    def resample1(self):
        t = np.argsort(self.weights)[:10]
        interesting_points = util.ht2xyp(self.particles[t])
        mean = np.mean(interesting_points , axis =0)
        cov = np.cov(interesting_points,rowvar =False)
        new_normal_particles = np.random.multivariate_normal(mean, cov,100)
        

        
        '''r = np.random.uniform(low=0.0, high=self.posrange, size=[int(self.count/10), 1])
        angle = np.random.uniform(low=-np.pi, high=np.pi, size=[int(self.count/10), 1])
        xy = np.sqrt(r) * np.hstack([np.cos(angle), np.sin(angle)])
        dxyp = np.hstack([xy, np.random.uniform(
            low=-self.angrange, high=self.angrange, size=[int(self.count/10), 1])])
        new_particles= []
        for i in range(10):
            new_particles.append(np.matmul(interesting_points[i], util.xyp2ht(dxyp)))
        self.particles = list(np.asarray(new_particles)) 
        self.particles = np.concatenate(self.particles, axis = 0)'''
        self.particles = util.xyp2ht(new_normal_particles)
        self.weights[:] = 1.0 / self.count
