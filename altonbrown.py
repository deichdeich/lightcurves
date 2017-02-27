"""
Author: Alex Deich
Date: Dec, 2016

altonbrown.py: spits out the coordinates of an equal-arrival-time surface (EATS).
good_eats(**kwargs) returns a 3-column np.array with cols for r,theta,phi.  NB: EATS is
symmetric about line-of-sight, in this case taken to be phi.

Usage:
good_eats(G_sh: the lorentz factor of the shock
          t: the lab time in seconds
          r_dec: a nondimensionalization parameter for distance
          numbins: the spatial resolution of the surface
          alpha: external gas density parameter
          delta: radiative or adiabatic evolution)



All equations from Panaitescu&Meszaros 1998, all units cgs.
"""
from __future__ import division, print_function
import numpy as np

cc = 29979245800. # speed of light in cm/s

def a(r,r_dec):
    return(r / r_dec)

def n(alpha, delta):
    ret = (3 - alpha) / (1 + delta)
    return(ret)

def tau(t,t_dec):
    return(t / t_dec)

def theta_func(G_sh, t, t_dec, r, r_dec, alpha, delta):
    radical = (tau(t,t_dec)/a(r,r_dec)) - (a(r,r_dec)**(2*n(alpha, delta))/(2*n(alpha, delta) + 1))
    ret = 2 * np.arcsin((2 * G_sh)**(-1) * np.sqrt(radical))
    return(ret)

def get_rlim(G_sh, t, r_dec, alpha, delta):
    """
    The sqrt in the theta equation gives imaginary nums for r's after a limit set by
    r = 2 * r_dec * (t_dec / (t * (2n + 1))) ^ (-1 / (2n + 1))
    So you want a range of r's that go from 0 to this value.
    """
    t_dec = get_t_dec(G_sh, r_dec)
    r_lim = r_dec * ((2 * n(alpha, delta) + 1) * (t/t_dec)) ** (1/(2 * n(alpha, delta) + 1))
    return(r_lim)

def get_t_dec(G, r_dec):
    t_dec = r_dec / (2 * G ** 2 * cc)
    return(t_dec)
    
def good_eats(G_sh, t, r_dec, numbins, alpha, delta):
    t_dec = get_t_dec(G_sh, r_dec)
    
    # get the furthest radius    
    r_lim = get_rlim(G_sh, t, r_dec, alpha = 0, delta = 0)
    
    # r, theta values for one phi slice
    r_vals = np.linspace(0.0001, r_lim, numbins, endpoint=True)
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
    dphi = (2 * np.pi) / numbins
    for phi in np.linspace(0, 2 * np.pi, numbins):
        surface[lower_bound:upper_bound] = phi_slice
        surface[lower_bound:upper_bound,2] = phi
        lower_bound += numbins
        upper_bound += numbins
    
    # np.arcsin is not happy with r close to 0.  This takes any NaN value and replaces
    # it with 0.
    surface = np.nan_to_num(surface)
    
    return(surface)
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from time import time
    start = time()
    test_eats = good_eats(G_sh = 1e4,
                          t = 100,
                          r_dec = 1e16,
                          numbins = 600,
                          alpha = 0,
                          delta = 0)
    end = time()
    print("Computation time:", end-start)

    #3d test plot
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    x = test_eats[:,0] * np.sin(test_eats[:,1]) * np.cos(test_eats[:,2])
    y = test_eats[:,0] * np.cos(test_eats[:,1])
    z = test_eats[:,0] * np.sin(test_eats[:,1]) * np.sin(test_eats[:,2])
    ax.plot(x, y, z, color = "blue", alpha = 0.1)       
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.zaxis.set_major_formatter(plt.NullFormatter())

       
    plt.show()
    
#    uncomment these lines to write the surface out to a CSV file.
#    print("Writing to file...")
#    np.savetxt("test_eats.csv", test_eats) 