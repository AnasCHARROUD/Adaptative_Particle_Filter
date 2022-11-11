#!/usr/bin/env python

import os

import numpy
import pykitti

#import kittidrives
import util


class kittiwrapper:
    def __init__(self, datadir):
        self.ododir = os.path.join(datadir, 'odometry')
        self.rawdir = datadir

    def sequence(self, i,date,drive):
        if 0 <= i < 11:
            sequence = pykitti.odometry(self.ododir, '{:02d}'.format(i))
            sequence.poses = numpy.array(sequence.poses)
        else:
            date = date
            drive = drive
            sequence = pykitti.raw(self.rawdir, date, drive)
            T_imu_cam0 = util.invert_ht(sequence.calib.T_cam0_imu)
            sequence.poses = numpy.array(
                [numpy.matmul(oxts.T_w_imu, T_imu_cam0) \
                    for oxts in sequence.oxts])
        return sequence
