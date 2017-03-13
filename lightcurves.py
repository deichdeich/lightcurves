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
import altonbrown
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
        self.G_c = 1e4 # eq. 2.  Lorentz factor at theta = 0
        self.al_g = 0  # eq. 2.  shape of the Lorentz distribution in the wings
        self.be_g = 1  # eq. 2.  smoothness between jet core and wings
        self.T_j = 1  #  Jet half-opening angle
        self.dt = dt # Timestepping interval
        self.t_lab = t_lab
        self.epsilon_B = 0.005 # in the equation for B
        self.epsilon_e = 0.01 # eq. 13.  The fraction of kinetic energy given to electrons
        self.m_p = 1.6726219e-24 # eq. 13. proton mass
        self.m_e = 9.10938e-28 # eq. 13. electron mass
        self.e = 1.60217662e-19 # electron charge
        self.n = 0.1 # eq. 18. number density
        self.E_iso = 1e53 # isotropic equivalent luminosity, E_iso = 4 * pi * epsilon
                                                    # for energy-per-solid angle epsilon
       
        self.sigma_T = 6.6524e-25 # thomson cross section in cm^2               
        
        self.nu_obs = nu_obs # observation frequency                                      

        numbins = spatial_res
        self.e_c = self.E_iso/numbins
       
        self.dph = (2 * np.pi)/numbins
        self.numbins = numbins # the resolution of the surface.  At the moment, r, th, and
                                   # ph are all binned with this number.
        
        self.movie = movie
        
        # stuff that will be updated over the course of the integration
        self.eats = 999 # The EATS, which will be recalculated at each timestep
        self.G_j = self.G_c # jet Lorentz
        #self.G_sh = 1 + 1.4142135623730951 * (self.G_j - 1) # Lorentz factor of the shock.
                                                            # Does writing sqrt(2) like
                                                            # this speed up the calculation?
                                                            # who's to say?
        self.G_sh = self.G_j # vector or scalar? ask about it.        
        self.M_0_inv = 999 # (initial fireball rest mass)^-1 as a function of angle
        self.lightcurve = 999 # this will eventually have the array with entries for the intensity at each timestep.

    def energy_per_omega(self):
        return(self.e_c/(1 + (self.eats[:, 1]/self.T_c)**(self.al_e * self.be_e))**(1/self.be_e))
    
    def lorentz_per_omega(self):
        return(self.G_c/(1 + (self.eats[:, 1]/self.T_c)**(self.al_g * self.be_g))**(1/self.be_g))
    
    def initialize_M_0_inv(self):
        M_0_inv = self.lorentz_per_omega() * cc ** 2 / self.energy_per_omega()
        return(M_0_inv)
    
    def density(self):
        return(self.m_p * self.n)
    
    def update_gamma(self):
        gamma_0 = self.eats[:,4]
        f = self.eats[:,3]
        numerator = np.sqrt(1 + (4 * gamma_0 * f) + (4 * f**2)) - 1
        denominator = 2 * f
        gamma = numerator/denominator
        

        if gamma.any() < 1:
            raise ArithmeticError("Gamma less than 1!")
            
        return(gamma)
    
    def update_delta(self):
        gamma = self.eats[:,5]
        beta = beta_from_gamma(gamma)
        denominator = gamma * (1 - beta * np.cos(self.eats[:,1]))
        return(1/denominator)
        
    
    #eq. 5. ratio of swept-up mass to initial fireball rest mass, function of theta.        
    def update_f(self):
        omega = solid_angle(self.eats[:,1])
        rho = self.density()
        df = (self.M_0_inv * omega * rho * self.eats[:,0]**2 * self.dr)
        return(df)
    
    def update_B(self):
        const = np.sqrt(32 * np.pi * self.epsilon_B * self.m_p * cc**2 * self.n)
        otherthing = np.sqrt(self.eats[:,5]**2 - 1)
        
        return(const * otherthing)
    
    def update_g_m(self):
        return(self.m_p / self.m_e * self.epsilon_e * (self.eats[:,5] - 1) )
    
    def update_P(self):
        const = (4 / 3) * self.sigma_T * cc * (1 / (8 * np.pi))
        otherthing = self.eats[:,7]**2 * (self.eats[:,8]**2 - 1)
        return(const * otherthing)
    
    def update_nu_m(self):
        numerator = 0.25 * self.e * self.eats[:,8]**2 * self.eats[:,7]
        denominator = self.m_e * cc
        return(numerator/denominator)
    
    def update_I_p(self):
        numerator = self.eats[:,10] * self.n * self.eats[:,0]
        return(numerator/self.eats[:,9])
    
    def update_dL(self):
        return(self.eats[:,11] * self.eats[:,6]**3 * self.eats[:,0]**2 * np.sin(self.eats[:,1]) * self.dTh * self.dph)
        
    def update_nu(self):
        return(self.nu_obs / self.eats[:,6])

    def update_I_nu(self):
        
        where_is_nu_m_greater = np.where(self.eats[:,9] > self.eats[:,13])
        where_is_nu_m_less = np.where(self.eats[:,9] < self.eats[:,13])
        I_nu = np.zeros(len(self.eats))
        I_nu[where_is_nu_m_greater] = self.eats[:, 11][where_is_nu_m_greater] * (self.eats[:, 13][where_is_nu_m_greater] / self.eats[:, 9][where_is_nu_m_greater])**(1/3)
        I_nu[where_is_nu_m_less] = self.eats[:, 11][where_is_nu_m_less] * (self.eats[:, 13][where_is_nu_m_less] / self.eats[:, 9][where_is_nu_m_less])**(-(self.eats[:, 10][where_is_nu_m_less] - 1)/2)
        
        return(I_nu)
    
    def boost_intensity(self):
        return(self.eats[:, 14] * self.eats[:, 6]**3)
        
    
    def do_all_the_calcs(self, dark_eats):
        # I'll make an array full of 0's that I'll fill with all the new calculations
        bright_eats = np.copy(self.eats)
        
        # This is how much the radius of each bin has changed.
        self.dr = dark_eats[:, 0] - self.eats[:, 0]
        self.dTh = dark_eats[:, 1] - self.eats[:, 1]
        """
        The columns are
        
        0: r, radius
        1: theta, polar angle
        2: phi, azimuthal angle
        3: f, ratio of swept-up mass to initial rest mass
        4: G_0, initial Lorentz factor
        5: G_j, instantaneous Lorentz factor of the jet
        6: delta, relativistic doppler factor
        7: B, magnetic field
        8: g_m, Lorentz factor of the matter
        9: nu_m, peak synchrotron frequency
        10: P, total power
        11: I_p, peak co-moving intensity
        12: dL, local observed luminosity
        13: nu, boosted co-moving frequency of nu_obs
        14: I_nu, co-moving intensity at observation frequency
        15: I_nu_obs, observed frequency intensity boosted to detector frame
        16: dL_nu, luminosity of observed frequency
        """
        # The order that these are performed does matter
        bright_eats[:, 0:3] = dark_eats
        bright_eats[:, 3] = bright_eats[:, 3] + self.update_f()
        bright_eats[:, 4] = self.eats[:,4]
        bright_eats[:, 5] = self.update_gamma()
        bright_eats[:, 6] = self.update_delta()
        bright_eats[:, 7] = self.update_B()
        bright_eats[:, 8] = self.update_g_m()
        bright_eats[:, 9] = self.update_nu_m()
        bright_eats[:, 10] = self.update_P()
        bright_eats[:, 11] = self.update_I_p()
        bright_eats[:, 12] = self.update_dL()
        bright_eats[:, 13] = self.update_nu()
        bright_eats[:, 14] = self.update_I_nu()
        bright_eats[:, 15] = self.boost_intensity()
        bright_eats[:, 16] = bright_eats[:, 15] * bright_eats[:, 0] * np.sin(bright_eats[:, 1]) * self.dTh * self.dph
        
        return(bright_eats)     
        
    def add_luminosity(self, step):
        # sum all of the luminosity bins and put them in the lightcurve array
        self.lightcurve[step,0] = np.nansum(self.eats[:,16][np.isfinite(self.eats[:,16])])            
        # put the current timestep in the other column of the lightcurve array
        #self.lightcurve[step,0] = np.nanmean(self.eats[:, 16])
        self.lightcurve[step,1] = self.t_lab            

    def time_evolve(self, nsteps):
        # for every timestep, pass the output of altonbrown() to
        # do_all_the_calcs(), whose output is all of the data for the current
        # timestep.  Then, sum the intensity column and record that sum in the
        # lightcurve array.
        nsteps = int(nsteps)
        self.lightcurve = np.zeros((nsteps,2))
        init_dark_eats = altonbrown.good_eats(G_sh = 1e4,
                                              t = self.t_lab,
                                              r_dec = 1e16,
                                              numbins = self.numbins,
                                              alpha = 0,
                                              delta = 0)
                                              
        self.eats = np.zeros((len(init_dark_eats), 17))
        self.eats[:,0:3] = init_dark_eats
        self.eats[:, 4] = self.lorentz_per_omega()
        self.eats[:, 5] = self.lorentz_per_omega()
        self.M_0_inv = self.initialize_M_0_inv()
        # this is where I store how long the last 200 timesteps took in order to
        # estimate time remaining
        times = np.zeros(400)
        
        for step in xrange(1,nsteps):
            
            # start the clock to time the calculation to estimate how long it'll take
            t1 = time()
            
            
            new_eats = altonbrown.good_eats(G_sh = self.eats[0,4],
                                            t = self.t_lab,
                                            r_dec = 1e16,
                                            numbins = self.numbins,
                                            alpha = 0,
                                            delta = 0)
                                            
            self.eats = self.do_all_the_calcs(new_eats)


            # add the timestep to the lightcurve
            self.add_luminosity(step)
            
            # advance the timestep
            self.t_lab += self.dt

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
        if step % 100 == 0:
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
        time = self.lightcurve[:,1]/86400
        luminosity = self.lightcurve[:,0]
        
        if ax == False:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        ax.loglog(time, luminosity, linewidth = 2)
        ax.set_xlim(xmin = time[10])
        ax.set_xlabel('t (days)')
        ax.set_ylabel(r'$L_\nu$ for $\nu = 10^{{{}}}$Hz'.format(np.int(np.log10(self.nu_obs))))
        
        if savefig == True:
            plt.savefig(fname)
        elif savefig == False:
            plt.show()
        
        if ax == False:
            plt.close()

    def plot_3d_heatmap(self, savefig = False, fname = "heatmap.png", ax = False):         
        # Here are the coordinates of the things I'm going to plot.  "loglum" is the
        # log of the luminosity, which is how I color the points.
        x = self.eats[:,0] * np.sin(self.eats[:,1]) * np.cos(self.eats[:,2])
        y = self.eats[:,0] * np.cos(self.eats[:,1])
        z = self.eats[:,0] * np.sin(self.eats[:,1]) * np.sin(self.eats[:,2])
        loglum = np.log10(np.nan_to_num(self.eats[:,12]))
        
        # If there is no axis to plot to given in the arguments, then make your own:
        if ax == False:
            fig = plt.figure()
            ax = fig.gca(projection = '3d')
        
            p = ax.scatter(x, y, z, c = loglum, vmin = 30, vmax = 41, alpha = .01, edgecolor = 'none')        
            cb = plt.colorbar(p)
            cb.set_clim(30,41)
            cb.set_alpha(1)
            cb.draw_all()
            cb.set_label(r'$\mathrm{luminosity} \quad \left(\mathrm{erg}\> \mathrm{s}^{-1} \mathrm{cm}^{-2}\right)$')
            cb.set_ticks([30,32,34,36,38,40])
            cb.set_ticklabels([r"$10^{30}$",r"$10^{32}$",r"$10^{34}$",r"$10^{36}$",r"$10^{38}$",r"$10^{40}$"])
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.zaxis.set_major_formatter(plt.NullFormatter())
            #ax.set_zlim(-3e13,3e13)
            ax.yaxis.set_major_formatter(plt.NullFormatter())
            #ax.set_xlim(-3e13,3e13)
            #ax.set_ylim(0,3e16)
            ax.view_init(30,30)
            ax.set_ylabel(r'$10^{16}\mathrm{cm}$', labelpad = -10)
            ax.set_xlabel(r'$10^{13}\mathrm{cm}$', labelpad = -10)
            ax.set_zlabel(r'$10^{13}\mathrm{cm}$', labelpad = -10)
            
            if savefig == False:
                plt.show()
            elif savefig == True:
                plt.savefig(fname)
            plt.close()

        else:
            p = ax.scatter(x, y, z, c = loglum, vmin = 30, vmax = 41, alpha = .01, edgecolor = 'none')     
            cb = plt.colorbar(p, ax = ax)
            cb.set_clim(28,35)
            cb.set_alpha(1)
            cb.draw_all()
            cb.set_label(r'$\log{L} \quad \left(\mathrm{erg}\> \mathrm{s}^{-1} \mathrm{cm}^{-2}\right)$')
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.NullFormatter())
            ax.zaxis.set_major_formatter(plt.NullFormatter())
            ax.set_zlim(-6e15,6e15)
            ax.set_xlim(-6e15,6e15)
            ax.set_ylim(0,1.5e17)
            ax.view_init(30,30)
            ax.set_ylabel(r'$10^{17}\mathrm{cm}$', labelpad = -15)
            
            if savefig == False:
                plt.show()
            elif savefig == True:
                plt.savefig(fname)
            plt.close()

    def plot_both(self, savefig = False, fname = "comparison.png"):
        fig = plt.figure()
        heatmap_axis = fig.add_subplot(122, projection = '3d', aspect = 0.1)
        lightcurve_axis = fig.add_subplot(121, aspect = 1.3)
        self.plot_3d_heatmap(ax = heatmap_axis)
        self.plot_lightcurve(ax = lightcurve_axis)
        if savefig == False:
            plt.show()
        elif savefig == True:
            plt.savefig(fname)
        
        plt.close()


                    
if __name__ == "__main__":
    test_curve = Lightcurve(spatial_res = 10, dt = 1, t_lab = 1e3)#, movie = "heatmap")
    test_curve.time_evolve(nsteps = 1e6)
    #test_curve.plot_3d_heatmap(savefig = True)
    test_curve.plot_lightcurve()
    #test_curve.plot_both()
    #print("\n",test_curve.lightcurve)
        
        
        
        
        
        
        
        
        
        
        
        
        
        