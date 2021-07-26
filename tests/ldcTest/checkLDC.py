#!/usr/bin/python

#############################################################################################################################################
 # Saras
 # 
 # Copyright (C) 2019, Mahendra K. Verma
 #
 # All rights reserved.
 # 
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions are met:
 #     1. Redistributions of source code must retain the above copyright
 #        notice, this list of conditions and the following disclaimer.
 #     2. Redistributions in binary form must reproduce the above copyright
 #        notice, this list of conditions and the following disclaimer in the
 #        documentation and/or other materials provided with the distribution.
 #     3. Neither the name of the copyright holder nor the
 #        names of its contributors may be used to endorse or promote products
 #        derived from this software without specific prior written permission.
 # 
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 # ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 # ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 # (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 # LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 # ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 #
 ############################################################################################################################################
 ##
 ##! \file checkLDC.py
 #
 #   \brief Python script to validate SARAS with results of Ghia et al
 #
 #   \author Roshan Samuel
 #   \date Jan 2020
 #   \copyright New BSD License
 #
 ############################################################################################################################################
 ##

ptFile = True

if ptFile:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

from scipy.interpolate import griddata
import numpy as np
import h5py as hp
import yaml as yl

# Pyplot-specific directives
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams["font.weight"] = "medium"

def init():
    global U, W
    global Nx, Nz
    global figSize
    global xLen, zLen

    Nx = 0
    Nz = 0
    xLen = 0.0
    zLen = 0.0

    U = np.zeros([1, 1, 1])
    W = np.zeros([1, 1, 1])

    figSize = (12, 6)


def parseYAML(paraFile):
    global Nx, Nz
    global xLen, yLen, zLen

    yamlFile = open(paraFile, 'r')
    yamlData = yl.load(yamlFile)

    Nx = 2**yamlData["Mesh"]["X Index"] + 1
    Nz = 2**yamlData["Mesh"]["Z Index"] + 1

    xLen = yamlData["Program"]["X Length"]
    zLen = yamlData["Program"]["Z Length"]


def loadGhia():
    global u_ghia, v_ghia

    u_ghia = np.loadtxt("u_profile_ghia.dat", comments='#')
    v_ghia = np.loadtxt("v_profile_ghia.dat", comments='#')


def loadData(timeVal):
    global Nx, Nz
    global U, W
    global X, Z

    fileName = "output/Soln_{0:09.4f}.h5".format(float(timeVal))

    try:
        f = hp.File(fileName, 'r')
    except:
        print("Could not open file " + fileName + "\n")
        exit()

    # Initialize and read staggered grid data
    U = np.array(f['Vx'])
    W = np.array(f['Vz'])

    X = np.array(f['X'])
    Z = np.array(f['Z'])


def plotProfile():
    global X, Z
    global U, W
    global Nx, Nz
    global u_ghia, v_ghia

    if ptFile:
        plt.switch_backend('agg')

    # Plot a data frame
    fig, axes = plt.subplots(1, 2, figsize=figSize)

    uProfile = (U[int(Nx/2), :] + U[int(Nx/2) - 1, :])/2
    axes[0].plot(u_ghia[:,2], u_ghia[:,1], marker='*', markersize=10, linestyle=' ', label='Ghia et al')
    axes[0].plot(uProfile, Z, linewidth=2, label='SARAS')
    axes[0].set_xlim([-0.6, 1.1])
    axes[0].set_xlabel(r"$u_x$", fontsize=25)
    axes[0].set_ylabel(r"$z$", fontsize=25)
    axes[0].tick_params(labelsize=20)
    axes[0].legend(fontsize=20)
    axes[0].set_title(r"$u_x$ at $x=0.5$", fontsize=25)

    vProfile = (W[:, int(Nz/2)] + W[:, int(Nz/2) - 1])/2
    axes[1].plot(v_ghia[:,1], v_ghia[:,2], marker='*', markersize=10, linestyle=' ', label='Ghia et al')
    axes[1].plot(X, vProfile, linewidth=2, label='SARAS')
    axes[1].set_ylim([-0.62, 0.42])
    axes[1].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[1].set_xlabel(r"$x$", fontsize=25)
    axes[1].set_ylabel(r"$u_z$", fontsize=25)
    axes[1].tick_params(labelsize=20)
    axes[1].legend(fontsize=20)
    axes[1].set_title(r"$u_z$ at $z=0.5$", fontsize=25)

    plt.gca().set_aspect('auto')
    plt.tight_layout()

    if ptFile:
        plt.savefig("ldc_validation.png")
    else:
        plt.savefig("ldc_validation.png")
        plt.show()


def checkTolerance():
    global X, Z
    global U, W
    global Nx, Nz
    global u_ghia, v_ghia

    uProfile = (U[int(Nx/2), :] + U[int(Nx/2) - 1, :])/2
    intpAxis = u_ghia[:,1]
    intpData = griddata(Z, uProfile, intpAxis)
    intpData[0] = 1.0;      intpData[-1] = 0.0

    avgError = np.mean(np.absolute(u_ghia[:,2] - intpData))
    avgValue = np.mean(np.absolute(u_ghia[:,2]))

    print("")
    print(r"Average absolute value of horizontal velocity, Ux = " + str(avgValue) + "\n")
    print("Average absolute error = " + str(avgError) + "\n")

    vProfile = (W[:, int(Nz/2)] + W[:, int(Nz/2) - 1])/2
    intpAxis = v_ghia[:,1]
    intpData = griddata(X, vProfile, intpAxis)
    intpData[0] = 0.0;      intpData[-1] = 0.0

    avgError = np.mean(np.absolute(v_ghia[:,2] - intpData))
    avgValue = np.mean(np.absolute(v_ghia[:,2]))

    print(r"Average absolute value of vertical velocity, Uz = " + str(avgValue) + "\n")
    print("Average absolute error = " + str(avgError) + "\n")


if __name__ == "__main__":
    init()

    parseYAML("input/parameters.yaml")

    loadData(30.0)

    loadGhia()

    plotProfile()

    checkTolerance()

