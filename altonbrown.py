"""
Author: Alex Deich
Date: Dec, 2016

altonbrown.py: spits out the coordinates of an equal-arrival-time surface (EATS).
good_eats(**kwargs) returns a 3x1 np.array with cols for r,theta,phi.  NB: EATS is
symmetric about line-of-sight, in this case taken to be phi.

Usage:
good_eats(G_sh: the lorentz factor of the shock,
          t: the lab time in seconds,
          t_dec: a nondimensionalization parameter for time,
          r_dec: a nondimensionalization parameter for distance,
          r_lim: the maximum distance to which to calculate the EATS; you can only go so far before getting imaginary numbers in the radical,
          numbins: the spatial resolution of the surface; the higher this is, the higher r_lim can be,
          alpha: external gas density parameter,
          delta: radiative or adiabatic evolution)

All equations from Panaitescu&Meszaros 1998
"""
from __future__ import division, print_function
import numpy as np

cc = 29979245800. # speed of light in cm/s

def a(r,r_dec):
    return(r/r_dec)

def n(alpha, delta):
    ret = (3-alpha)/(1+delta)
    return(ret)

def tau(t,t_dec):
    return(t/t_dec)

def theta_func(G_sh, t, t_dec, r, r_dec, alpha, delta):

    radical = (tau(t,t_dec)/a(r,r_dec)) - (a(r,r_dec)**(2*n(alpha, delta))/(2*n(alpha, delta) + 1))
    ret = 2 * np.arcsin( (2 * G_sh)**(-1) * np.sqrt(radical))
    return(ret)

def get_rlim(G_sh, t, r_dec, alpha, delta):
    
    """
    The radical in the theta equation gives NaN's for r's after a limit set by
    r = r_dec * (t_dec / (t * (2n + 1))) ^ (-1 / (2n + 1))
    So you want a range of r's that go from 0 to this value.
    Don't know why you need to double it.  Seems to be exactly right though
    """
    t_dec = get_t_dec(G_sh, r_dec)
    r_lim = 2 * r_dec * ((2 * n(alpha, delta) + 1) * (t/t_dec)) ** (1/(2 * n(alpha, delta) + 1))
    return(r_lim)

def get_t_dec(G, r_dec):
    t_dec = r_dec/(2 * G ** 2 * cc)
    return(t_dec)
    
def good_eats(G_sh, t, r_dec, r_lim, numbins, alpha, delta):
    
    t_dec = get_t_dec(G_sh, r_dec)
    
    #print("t_dec:", t_dec)
    
    # r, theta values for one phi slice
    r_vals = np.linspace(0.0001, r_lim-r_lim/2, numbins, endpoint=True)
    theta_vals = theta_func(G_sh, t, t_dec, r_vals, r_dec, alpha, delta)
    
    # a dummy array to fill with the right phi values
    phis = np.zeros_like(r_vals)
    
    # This makes one array of size (numbins, 3) with cols for
    # r, theta, phi (phi is still a dummy column)
    phi_slice = np.vstack([r_vals,theta_vals,phis]).T
    
    # This is the array which will hold the coordinates of the middle of each bin of the
    # surface.
    surface = np.zeros((numbins**2,3))

    # This fills the phi dummy columns.  Phi is taken over a range of equally-spaced
    # values from 0 to 2pi
    lower_bound = 0
    upper_bound = numbins
    dphi = (2 * np.pi)/numbins
    for phi in np.linspace(0, 2 * np.pi, numbins):
        surface[lower_bound:upper_bound] = phi_slice
        surface[lower_bound:upper_bound,2] = phi
        lower_bound += numbins
        upper_bound += numbins
    
    surface = np.nan_to_num(surface)
    
    return(surface)
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from time import time
    t1 = 100
    r_lim1 = get_rlim(G_sh = 1e4, t = t1, r_dec = 1e16, alpha = 0, delta = 0)
    print("r_lim:",r_lim1)
    start = time()
    test_eats = good_eats(G_sh = 1e4, t = t1,
                                      r_dec = 1e16,
                                      r_lim = r_lim1,
                                      numbins = 600,
                                      alpha = 0,
                                      delta = 0)
    end = time()
    print("Computation time:", end-start)

    """
    #2d test plot
    plt.plot(test_eats[:,0] * np.cos(test_eats[:,1]), test_eats[:,0] * np.sin(test_eats[:,1]))
    plt.show()
    """
    #3d test plot
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.plot(test_eats[:,0] * np.sin(test_eats[:,1]) * np.cos(test_eats[:,2]),
                    test_eats[:,0] * np.cos(test_eats[:,1]),
                    test_eats[:,0] * np.sin(test_eats[:,1]) * np.sin(test_eats[:,2]), color = "blue", alpha = 0.1)
    ax.plot(-1 * test_eats[:,0] * np.sin(test_eats[:,1]) * np.cos(test_eats[:,2]),
                    -1 * test_eats[:,0] * np.cos(test_eats[:,1]),
                    -1 * test_eats[:,0] * np.sin(test_eats[:,1]) * np.sin(test_eats[:,2]), color = "red", alpha = 0.05)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    #ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.zaxis.set_major_formatter(plt.NullFormatter())

       
    plt.show()
    """
    # wireframe test plot
    # matplotlib plot_surface is very slow.  Apparently the cool thing to use is MayaVi
    # but that was too difficult to install.
    r = test_eats[:,0]
    theta = test_eats[:,1]
    phi = test_eats[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = r * np.outer(np.cos(phi), np.sin(theta))
    y = r * np.outer(np.sin(phi), np.sin(theta))
    z = r * np.outer(np.ones(np.size(phi)),np.cos(theta))
    
    ax.plot_surface(x,y,z,color='b')
    plt.show()
    """
    """
    # trisurface test plot
    # also sucks
    r = test_eats[:,0]
    theta = test_eats[:,1]
    phi = test_eats[:,2]
    
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.plot_trisurf(x,y,z)
    plt.show()
    """
    
    
#    print("Writing to file...")
#    np.savetxt("test_eats.csv", test_eats) 