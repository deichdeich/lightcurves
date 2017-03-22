"""
Author: Alex Deich
Date: January, 2017

lightcurves.py: Calculates GRB lightcurves after the prescription in Rossi et al. 2002.

Requires altonbrown.py to generate equal arrival time surface.

All units cgs.
"""

from __future__ import division, print_function
import sys
import numpy as np
import altonbrown as altonbrown
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time

### Defining global constants
cc = 29979245800 # speed of light in cm/s

### Some helpful functions
def gamma_from_beta(beta):
    return(np.sqrt(1/(1 - beta**2)))

def beta_from_gamma(gamma):
    return(np.sqrt(1 - (1/gamma**2)))

def solid_angle(theta):
    return(4 * np.pi * (1 - np.cos(theta)))

def surface_area(r, theta, dTh, dph):
    return((r**2) * np.sin(theta) * dTh * dph)

# maybe output a dictionary?  want unique mass for each theta
def mass_from_energy(energy, lorentz):
    return(energy/(lorentz * cc**2))
    

###################################################

### Calculate, time-evolve and plot luminosity ###

###################################################
class Lightcurve(object):
    def __init__(self, spatial_res, dt = 0.0001, t_lab = 0, nu_obs  = 1e15, movie = False):
        
        # initializing constants (there should be a lot)
        self.T_c = 90  # eq. 1.  core angular size
        self.al_e = 0  # eq. 1.  shape of the energy distribution in the wings
        self.be_e = 1  # eq. 1.  smoothness between jet core and wings
        self.G_sh = 1e4 # eq. 2.  Lorentz factor of the jet at theta = 0
        self.al_g = 0  # eq. 2.  shape of the Lorentz distribution in the wings
        self.be_g = 1  # eq. 2.  smoothness between jet core and wings
        self.T_j = 1  #  Jet half-opening angle
        self.dt = dt # Timestepping interval
        self.t_lab = t_lab
        self.t_lab0 = t_lab
        self.t_exp = 0
        self.epsilon_B = 0.005 # in the equation for B
        self.epsilon_e = 0.01 # eq. 13.  The fraction of kinetic energy given to electrons
        self.m_p = 1.6726219e-24 # eq. 13. proton mass
        self.m_e = 9.10938e-28 # eq. 13. electron mass
        self.e = 4.80320427e-10 # electron charge
        self.n = 0.1 # eq. 18. number density
        self.E_iso = 1e53 # isotropic equivalent energy, E_iso = 4 * pi * epsilon
                                                    # for energy-per-solid angle epsilon
       
        self.sigma_T = 6.6524e-25 # thomson cross section in cm^2               
        
        self.nu_obs = nu_obs # observation frequency                                      

        numbins = spatial_res
        self.e_c = self.E_iso/(numbins**2)
        self.G_c = self.G_sh/(numbins**2)
        self.dph = (2 * np.pi)/numbins
        self.numbins = numbins # the resolution of the surface.  At the moment, r, th, and
                                   # ph are all binned with this number.
        
        self.movie = movie
        
        # colnames is the name and datatype of everything that will be updated.
        self.colnames = [('r', 'float'),  # r, radius
                         ('Th', 'float'), # theta, polar angle
                         ('Ph', 'float'), # phi, azimuthal angle
                         ('f', 'float'), # f, ratio of swept-up mass to initial rest mass
                         ('G_0', 'float'), # G_0, initial Lorentz factor
                         ('G_j', 'float'), # G_j, instantaneous Lorentz factor of the jet
                         ('delta', 'float'), # delta, relativistic doppler factor
                         ('B', 'float'), # B, magnetic field
                         ('g_m', 'float'), # g_m, Lorentz factor of the matter
                         ('nu_m', 'float'), # nu_m, peak synchrotron frequency
                         ('P', 'float'), # P, total power
                         ('I_p', 'float'), # I_p, peak co-moving intensity
                         ('dL', 'float'), # dL, local observed luminosity
                         ('nu', 'float'), # nu, boosted co-moving frequency of nu_obs
                         ('I_nu', 'float'), # I_nu, co-moving intensity at observation frequency
                         ('I_nu_obs', 'float'), # I_nu_obs, observed frequency intensity boosted to detector frame
                         ('dL_nu', 'float')] # dL_nu, luminosity of observed frequency

        # stuff that will be updated over the course of the integration
        self.eats = np.nan # The EATS, which will be recalculated at each timestep
        self.G_j = self.G_sh # jet Lorentz
        #self.G_sh = 1 + 1.4142135623730951 * (self.G_j - 1) # Lorentz factor of the shock.
                                                            # Does writing sqrt(2) like
                                                            # this speed up the calculation?
                                                            # who's to say?
        self.M_0_inv = np.nan # (initial fireball rest mass)^-1 as a function of angle
        self.lightcurve = np.nan # this will eventually have the array with entries for the intensity at each timestep.
    
    def get_r_dec(self, E_iso, n, m_p, gamma):
        denominator = gamma**2 * cc**2 * 4 * np.pi * m_p * n
        numerator = 3 * E_iso
        r_dec = (numerator/denominator)**(1/3)
        if np.isnan(r_dec):
            raise ValueError('r_dec is undefined')
        return(r_dec)   
        
    def energy_per_omega(self, theta):
        return(2 * self.e_c/(1 + (theta/self.T_c)**(self.al_e * self.be_e))**(1/self.be_e))
    
    def lorentz_per_omega(self, theta):
        return(2 * self.G_c/(1 + (theta/self.T_c)**(self.al_g * self.be_g))**(1/self.be_g))
    
    def initialize_M_0_inv(self, lorentz_per_omega, energy_per_omega):
        M_0_inv = (lorentz_per_omega * cc ** 2) / energy_per_omega
        return(M_0_inv)
    
    def density(self):
        return(self.m_p * self.n)
    
    def update_gamma(self, gamma_0, f):
        numerator = np.sqrt(1 + (4 * gamma_0 * f) + (4 * f**2)) - 1
        denominator = 2 * f
        gamma = numerator/denominator
        
        # check for superluminal behavior
        if gamma.any() < 1:
            raise ArithmeticError("Gamma less than 1!")
            
        return(gamma)
    
    def update_delta(self, theta, gamma):
        beta = beta_from_gamma(gamma)
        denominator = gamma * (1 - beta * np.cos(theta))
        return(1/denominator)
        
    
    def update_f(self, r, theta):
        omega = solid_angle(theta)
        rho = self.density()
        df = (self.M_0_inv * omega * rho * r**2 * self.dr)
        return(df)
    
    def update_B(self, G_j):
        const = np.sqrt(32 * np.pi * self.epsilon_B * self.m_p * cc**2 * self.n)
        otherthing = np.sqrt(G_j**2 - 1)
        
        return(const * otherthing)
    
    def update_g_m(self, G_j):
        return(self.m_p / self.m_e * self.epsilon_e * (G_j - 1) )
    
    def update_P(self, B, g_m):
        const = (4 / 3) * self.sigma_T * cc * (1 / (8 * np.pi))
        otherthing = B**2 * (g_m**2 - 1)
        return(const * otherthing)
    
    def update_nu_m(self, B, g_m):
        numerator = 0.25 * self.e * g_m**2 * B
        denominator = self.m_e * cc
        return(numerator/denominator)
    
    def update_I_p(self, r, nu_m, P):
        numerator = P * self.n * r
        return(numerator/nu_m)
    
    def update_dL(self, r, theta, I_p, delta):
        return(I_p * delta**3 * r**2 * np.sin(theta) * self.dTh * self.dph)
        
    def update_nu(self, delta):
        return(self.nu_obs / delta)

    def update_I_nu(self, nu_m, nu, I_p, P): 
        
        I_nu = np.zeros(len(self.eats))
        
        for i in xrange(len(I_p)):
            if nu_m[i] > nu[i]:
                I_nu[i] = I_p[i] * (nu[i] / nu_m[i])**(1/3)
            elif nu_m[i] < nu[i]:
                I_nu[i] = I_p[i] * (nu[i] / nu_m[i])**((P[i] - 1)/2)
        
        return(I_nu)
    
    def boost_intensity(self, I, delta):
        return(I * delta**3)
    
    def do_all_the_calcs(self, dark_eats):
        # "bright_eats" is the new surface for which all the values are being calculated.
        # "dark_eats" is just the coordinate array from altonbrown.py
        
        # initialize bright_eats by copying the existing eats.
        bright_eats = np.copy(self.eats)
        
        # This is how much the radius of each bin has changed.
        self.dr = dark_eats['r'] - self.eats['r']
        self.dTh = dark_eats['Th'] - self.eats['Th']

        # The order that these are performed does matter
        bright_eats['r'] = dark_eats['r']
        
        bright_eats['Th'] = dark_eats['Th']
        
        bright_eats['Ph'] = dark_eats['Ph']
        
        bright_eats['f'] = self.eats['f'] + self.update_f(bright_eats['r'], bright_eats['Th'])                     
        
        bright_eats['G_0'] = self.eats['G_0']
        
        bright_eats['G_j'] = self.update_gamma(bright_eats['G_0'], bright_eats['f'])                               
        
        bright_eats['delta'] = self.update_delta(bright_eats['Th'], bright_eats['G_j'])                         
        
        bright_eats['B'] = self.update_B(bright_eats['G_j'])
        
        bright_eats['g_m'] = self.update_g_m(bright_eats['G_j'])
        
        bright_eats['nu_m'] = self.update_nu_m(bright_eats['B'], bright_eats['g_m'])
        
        bright_eats['P'] = self.update_P(bright_eats['B'], bright_eats['g_m'])
        
        bright_eats['I_p'] = self.update_I_p(bright_eats['r'],
                                             bright_eats['nu_m'],
                                             bright_eats['P'])
        
        bright_eats['dL'] = self.update_dL(bright_eats['r'],
                                           bright_eats['Th'],
                                           bright_eats['I_p'],
                                           bright_eats['delta'])
        
        bright_eats['nu'] = self.update_nu(bright_eats['delta'])

        bright_eats['I_nu'] = self.update_I_nu(bright_eats['nu_m'],
                                               bright_eats['nu'],
                                               bright_eats['I_p'],
                                               bright_eats['P'])
        
        bright_eats['I_nu_obs'] = self.boost_intensity(bright_eats['I_nu'],
                                                       bright_eats['delta'])
        
        bright_eats['dL_nu'] = self.update_dL(bright_eats['r'],
                                              bright_eats['Th'],
                                              bright_eats['I_nu'],
                                              bright_eats['delta'])
                
        return(bright_eats)     
        
    def add_luminosity(self, step, col = 'I_p'):
        # sum all of the luminosity bins and put them in the lightcurve array
        self.lightcurve[step,0] = self.t_lab           
        # put the current timestep in the other column of the lightcurve array
        self.lightcurve[step,1] = np.nansum(self.eats[col][np.isfinite(self.eats[col])])            

    def time_evolve(self, nsteps, status = True):
        # for every timestep, pass the output of altonbrown() to
        # do_all_the_calcs(), whose output is all of the data for the current
        # timestep.  Then, sum the intensity column and record that sum in the
        # lightcurve array.
        nsteps = int(nsteps)
        self.lightcurve = np.zeros((nsteps,2))
        init_r_dec = self.get_r_dec(self.E_iso, self.n, self.m_p, self.G_c)
        init_dark_eats = altonbrown.good_eats(G_sh = self.G_c,
                                              t = self.t_lab,
                                              r_dec = init_r_dec,
                                              numbins = self.numbins,
                                              alpha = 0,
                                              delta = 0)
        self.eats = np.zeros(len(init_dark_eats), dtype = self.colnames)
        self.eats['r'] = init_dark_eats['r']
        self.eats['Th'] = init_dark_eats['Th']
        self.eats['Ph'] = init_dark_eats['Ph']
        self.eats['G_0'] = self.lorentz_per_omega(self.eats['Th'])
        self.eats['G_j'] = self.lorentz_per_omega(self.eats['Th'])
        self.M_0_inv = self.initialize_M_0_inv(self.lorentz_per_omega(self.eats['Th']),
                                               self.energy_per_omega(self.eats['Th']))
        # this is where I store how long the last 200 timesteps took in order to
        # estimate time remaining
        times = np.zeros(400)
        for step in xrange(1,nsteps):
            
            # start the clock to time the calculation to estimate how long it'll take
            t1 = time()
            
            new_eats = altonbrown.good_eats(G_sh = self.eats['G_0'][0],
                                            t = self.t_lab,
                                            r_dec = init_r_dec,
                                            numbins = self.numbins,
                                            alpha = 0,
                                            delta = 0)
                                            
            self.eats = self.do_all_the_calcs(new_eats)


            # add the timestep to the lightcurve
            self.add_luminosity(step)
            
            # advance the timestep
            self.t_exp += self.dt
            self.t_lab = 10**self.t_exp
            self.t_lab += self.dt
            if status == True:
                # this is the progress text in terminal.
                sys.stdout.write("\r{}% of the integration complete, {}-ish minutes remaining".format((100 * step/nsteps),
                                                                                                      (np.round((np.mean(times[np.where(times!=0)]) * (nsteps - step))/60,
                                                                                                                decimals=1))))
                sys.stdout.flush()
            
            
            if self.movie is not False:
                self.movie_maker(step, nsteps)
            
            # stop the clock and add the time to the array
            t2 = time()
            times[step%400] = t2 - t1


    ####################################
    ###   Plotting and movie making  ###
    ####################################
    def movie_maker(self, step, nsteps):
        if step % 10 == 0:
            if self.movie == "comparison":
                self.plot_both(savefig = True,
                                     fname = "movies/comparison/comp_{}.png".format(step))
            
            elif self.movie == "heatmap":
                self.plot_3d_heatmap(savefig = True,
                                     fname = "movies/heatmap/heatmap_{}.png".format(step))
            
            elif self.movie == "lightcurve":
                self.plot_lightcurve(savefig = True,
                                     fname = "movies/lightcurve/lightcurve_{}.png".format(step))
                                         
            else:
                raise ValueError("{} is not a valid plot".format(self.movie))
            
    def plot_lightcurve(self, savefig = False, fname = "lightcurve.pdf", ax = False):        
        time = self.lightcurve[:,0]/86400
        luminosity = self.lightcurve[:,1]
        
        if ax == False:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        ax.loglog(time, luminosity, linewidth = 2)
        ax.set_xlim(xmin = time[0], xmax = 1e5)
        #ax.set_ylim(1e37,1e43)
        ax.set_xlabel('t (days)')
        ax.set_ylabel(r'$L_\nu$ for $\nu = {{{}}} $ Hz'.format(self.nu_obs))
        timestamp = "T = {:.2E} s".format(self.t_lab)
        ax.text(0,2.2, timestamp, transform=ax.transAxes, fontsize = 20)
        if savefig == False:
            #plt.show()
            pass
        elif savefig == True:
            plt.savefig(fname)
            plt.close()

    def plot_3d_heatmap(self, savefig = False, fname = "heatmap.png", ax = False):         
        # Here are the coordinates of the things I'm going to plot.  "loglum" is the
        # log of the luminosity, which is how I color the points.
        x = self.eats['r'] * np.sin(self.eats['Th']) * np.cos(self.eats['Ph'])
        y = self.eats['r'] * np.cos(self.eats['Th'])
        z = self.eats['r'] * np.sin(self.eats['Th']) * np.sin(self.eats['Ph'])
        loglum = np.log10(np.nan_to_num(self.eats['dL']))
        
        # If there is no axis to plot to given in the arguments, then make your own:
        if ax == False:
            fig = plt.figure()
            ax = fig.gca(projection = '3d')
            p = ax.scatter(x, y, z, c = loglum, edgecolor = 'none')   
            cb = plt.colorbar(p, ax = ax)
            cb.set_clim(35,41)
            cb.set_alpha(1)
            cb.draw_all()  
            ax.set_zlim(-1e15,1e15)
            ax.set_xlim(-1e15,1e15)
            ax.set_ylim(0,1.5e17)
            ax.view_init(30,30)
            
            if savefig == False:
                plt.show()
            elif savefig == True and ax == False:
                plt.savefig(fname)
            plt.close()

        else:
            p = ax.scatter(x, y, z, c = loglum, edgecolor = 'none')   
            cb = plt.colorbar(p, ax = ax)
            cb.set_clim(18,30)
            cb.set_alpha(1)
            cb.draw_all()
            cb.set_label(r'$\log\left(dL_\nu\right)$')  
            #ax.xaxis.set_major_formatter(plt.NullFormatter())
            #ax.yaxis.set_major_formatter(plt.NullFormatter())
            #ax.zaxis.set_major_formatter(plt.NullFormatter())
            #ax.set_zlim(-1e18,1e18)
            #ax.set_xlim(-1e18,1e18)
            #ax.set_ylim(0,5e18)
            ax.view_init(30,30)


    def plot_both(self, savefig = False, fname = "comparison.png"):
        fig = plt.figure()
        heatmap_axis = fig.add_subplot(211, projection = '3d')#, aspect = .005)
        lightcurve_axis = fig.add_subplot(212)#, aspect = 1.3)
        self.plot_3d_heatmap(ax = heatmap_axis)
        self.plot_lightcurve(ax = lightcurve_axis)
        if savefig == False:
            plt.show()
        elif savefig == True:
            plt.savefig(fname)
        
        plt.close()


                    
if __name__ == "__main__":
    test_curve = Lightcurve(spatial_res = 40, dt = .01, t_lab = 0)
    test_curve.time_evolve(nsteps = 1e3)
    #test_curve.plot_3d_heatmap()
    test_curve.plot_lightcurve()
    plt.show()
    #test_curve.plot_both()
    #print("\n",test_curve.lightcurve)
        
        
        
        
        
        
        
        
        
        
        
        
        
        