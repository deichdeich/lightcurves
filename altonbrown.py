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

def theta_from_r(G_sh, t, t_dec, r, r_dec, alpha, delta):
    radical = (tau(t,t_dec)/a(r,r_dec)) - (a(r,r_dec)**(2*n(alpha, delta))/(2*n(alpha, delta) + 1))
    ret = 2 * np.arcsin((2 * G_sh)**(-1) * np.sqrt(radical))
    ret = np.nan_to_num(ret)
    return(ret)

def r_from_theta(r, r_dec, theta, t, t_dec, G_sh, alpha, beta):
    first_term = (2 * G_sh * np.sin(theta/2))
    second_term = (t / t_dec) / (r / r_dec)
    third_term = ((r / r_dec)**(2 * n(alpha, beta)))/(2 * n(alpha,beta) + 1)
    return(np.nan_to_num((first_term**2) - second_term + third_term))

def a_from_theta(a, theta, t, r_dec, G_sh, alpha, beta):
    tau = t / get_t_dec(G_sh, r_dec)
    first_term = (2 * G_sh * np.sin(theta/2))
    second_term = tau / a
    third_term = (a**(2 * n(alpha, beta)))/(2 * n(alpha,beta) + 1)
    return(np.nan_to_num((first_term**2) - second_term + third_term))   

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
    t_dec = r_dec / (2 * (G ** 2) * cc)
    return(t_dec)

def good_eats(G_sh, r_dec, t, numbins, alpha, delta, calculation_method = "analytical"):    
    # get the furthest radius    
    r_lim = get_rlim(G_sh, t, r_dec, alpha = 0, delta = 0)
    # r, theta values for one phi slice
    if calculation_method == "analytical":
        """
        This is the analytical method which returns theta values from a range of r values.
        Fast, but returns gibberish at large times because of the arcsin.
        """
        t_dec = get_t_dec(G_sh, r_dec)
        r_vals = np.linspace(0, r_lim, numbins, endpoint = True)
        theta_vals = theta_from_r(G_sh, t, t_dec, r_vals, r_dec, alpha, delta)
    
    elif calculation_method == "numerical":
        """
        This is the numerical root-finding method which returns r values from a range of
        theta values.
        Slow, but always returns correct values.  Also requires much higher resolution,
        especially at small times (t<1e3s or so), because the radius is very sensitive to
        small changes in theta.
        
        It's broken right now.  Here's the whole story:
        
        First, I was calculating the surface in terms of t and r, the physical,
        dimensionfull coordinates.  This was bad for two reasons:
        (1) the surface is out at r ~ 1e17cm and so this gives really tiny values for
        theta.  (2) following the radial coordinate coordinate will intersect the surface
        (and therefore have a zero) at two places:  the back side and the front side.
        The backside would catch it most of the time, so the front of the surface (the
        most important part of the surface for calculating the light curve) would be very
        sparse.
        
        Then, I switched to calculating it in tau and a, the nondimensionalized
        time and radius coordinates.  This fixed the two problems above because the
        surface only goes out to a = 25 or so, and starts near 0.  However, this required
        me to get tau and a from t, r, t_dec, and r_dec, and it turns out that the
        surface converges for a very narrow range of tau and a.  Additionally, the values
        of tau for which it converges is really, really sensitive to small changes in
        r_dec.  For a standard r_dec, the first value of tau that worked corresponded
        to t = 1e4 seconds, which is not very helpful for the actual light curve
        calculation.
        
        Then, *even if* I ignore those problems, and only calculate the light curve for a
        very small portion of the curve, starting at t= 1e4 seconds, the actual curve
        itself is aphysical, growing exponentially.  This is the problem for which
        I have the least idea how to fix, because the points seem identical to the
        analytical method, but the resulting light curve is crazy.
        
        So, what does work is the surface calculation for a small set of tau and a.  When
        you compare the output of the numerical method with the analytic method for these
        values, there's perfect correspondence.  Even better, at large times, (t > 1e7 s)
        the numerical surface looks great, while the analytic surface implodes like
        before.
        """
        theta_vals = np.linspace(0., np.pi, numbins, endpoint = False)
        r_vals = np.zeros_like(theta_vals)
        for i in xrange(len(theta_vals)):
            Th = theta_vals[i]
            r_vals[i] = optimize.brentq(a_from_theta,
                                        -r_lim,
                                        r_lim + r_lim/10,
                                        args = (Th,
                                                t,
                                                r_dec,
                                                G_sh,
                                                alpha,
                                                delta))
        #r_vals *= r_dec


    
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
    for i in xrange(1):
        start = time()
        test_eats = good_eats(G_sh = 1e4,
                              r_dec = 1.1673054268071324e16,
                              t = 10,
                              numbins = 100,
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
    plt.ylim(-1,5)
    plt.scatter(x,y, linewidth = 3)
    
    """
    #3d test plot
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    x = test_eats['r'] * np.sin(test_eats['Th']) * np.cos(test_eats['Ph'])
    y = test_eats['r'] * np.cos(test_eats['Th'])
    z = test_eats['r'] * np.sin(test_eats['Th']) * np.sin(test_eats['Ph'])
    ax.scatter(x, y, z, color = "blue", alpha = 0.1)       
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.zaxis.set_major_formatter(plt.NullFormatter())
    ax.view_init(30,45)
"""
    plt.show()

#    uncomment these lines to write the surface out to a CSV file.
#    print("Writing to file...")
#    np.savetxt("test_eats.csv", test_eats) 