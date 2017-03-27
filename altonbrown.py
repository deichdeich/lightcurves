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
from scipy import optimize

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
    ret = np.nan_to_num(ret)
    return(ret)

def r_from_theta(r, r_dec, theta, t, t_dec, G_sh, alpha, beta):
    first_term = (2 * G_sh * np.sin(theta/2))
    second_term = (t / t_dec) / (r / r_dec)
    third_term = ((r / r_dec)**(2 * n(alpha, beta)))/(2 * n(alpha,beta) + 1)
    return((first_term**2) - second_term + third_term)

def get_rlim(G_sh, t, r_dec, alpha, delta):
    """
    The sqrt in the theta equation gives imaginary nums for r's after a limit set by
    r = r_dec * (t_dec / (t * (2n + 1))) ^ (1 / (2n + 1))
    So you want a range of r's that go from 0 to this value.
    """
    t_dec = get_t_dec(G_sh, r_dec)
    r_lim = r_dec * ((2 * n(alpha, delta) + 1) * (t/t_dec)) ** (1/(2 * n(alpha, delta) + 1))
    return(r_lim)

def get_t_dec(G, r_dec):
    t_dec = r_dec / (2 * G ** 2 * cc)
    return(t_dec)

def good_eats(G_sh, t, r_dec, numbins, alpha, delta, calculation_method = "analytical"):
    t_dec = get_t_dec(G_sh, r_dec)
    
    # get the furthest radius    
    r_lim = get_rlim(G_sh, t, r_dec, alpha = 0, delta = 0)
    
    # r, theta values for one phi slice
    
    
    if calculation_method == "analytical":
        """
        This is the analytical method which returns theta values from a range of r values.
        Fast, but returns gibberish at large times because of the arcsin.
        """
        r_vals = np.linspace(0, r_lim, numbins, endpoint=True)
        theta_vals = theta_func(G_sh, t, t_dec, r_vals, r_dec, alpha, delta)
    
    elif calculation_method == "numerical":
        """
        This is the numerical root-finding method which returns r values from a range of
        theta values.
        Slow, but always returns correct values.  Also requires much higher resolution,
        especially at small times (t<1e3s or so), because the radius is very sensitive to
        small changes in theta.
        """
        theta_vals = np.linspace(0., np.pi, numbins, endpoint = False)
        r_vals = np.zeros_like(theta_vals)
        for i in xrange(len(theta_vals)):
            Th = theta_vals[i]
            r_vals[i] = optimize.brentq(r_from_theta,
                                        0.01,
                                        r_lim + r_lim/10,
                                        args = (r_dec,
                                                Th,
                                                t,
                                                t_dec,
                                                G_sh,
                                                alpha,
                                                delta))
    
    
    # This is the array which will hold the coordinates of the middle of each bin of the
    # surface.
    surface = np.zeros(numbins**2, dtype = [('r', 'float'),('Th','float'),('Ph','float')])

    # This fills the phi dummy columns.  Phi is taken over a range of equally-spaced
    # values from 0 to 2pi
    lower_bound = 0
    upper_bound = numbins
    dphi = (2 * np.pi) / numbins

    
    for phi in np.linspace(0, (2 * np.pi) - dphi, numbins):
        surface['r'][lower_bound:upper_bound] = r_vals
        surface['Th'][lower_bound:upper_bound] = theta_vals
        surface['Ph'][lower_bound:upper_bound] = phi
        lower_bound += numbins
        upper_bound += numbins

    return(surface)    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from time import time
    times = []
    for i in xrange(100):
        start = time()
        test_eats = good_eats(G_sh = 1e4,
                                  t = 10,
                                  r_dec = 1e16,
                                  numbins = 1000,
                                  alpha = 0,
                                  delta = 0,
                                  calculation_method = "numerical")
        end = time()
        times.append(end-start)
    print("Computation time:", np.mean(times))

    #phi slice test plot
    numbins = int(np.sqrt(len(test_eats['r'])))
    y = test_eats['r'][0:numbins] * np.sin(test_eats['Th'][0:numbins])
    x = test_eats['r'][0:numbins] * np.cos(test_eats['Th'][0:numbins])
    plt.plot(x,y, linewidth = 3)

    """
    #3d test plot
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    x = test_eats['r'] * np.sin(test_eats['Th']) * np.cos(test_eats['Ph'])
    y = test_eats['r'] * np.cos(test_eats['Th'])
    z = test_eats['r'] * np.sin(test_eats['Th']) * np.sin(test_eats['Ph'])
    ax.plot(x, y, z, color = "blue", alpha = 0.1)       
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.zaxis.set_major_formatter(plt.NullFormatter())
    ax.view_init(30,45)
    """

    plt.show()
 
#    uncomment these lines to write the surface out to a CSV file.
#    print("Writing to file...")
#    np.savetxt("test_eats.csv", test_eats) 