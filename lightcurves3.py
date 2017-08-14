"""
Author: Alex Deich
E-mail: alex.d.deich@gmail.com
Github: github.com/deichdeich
Date: July, 2017

Based on code by D. Lazzati at Oregon State University,
lazzatid@science.oregonstate.edu
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
####
yes, everything is a function of r and theta.  iterate through the possible values for
every thing that depends on energy and lorentz, including t_lab and R_EATS

make sure you can put in arbitrary arrays of energy-per-sa and lorentz-per-sa
####


### Some global constants ###
cc = 3e10  # speed of light in cm/s
m_p = 1.67e-24 # mass of proton
m_e = m_p / 1836. # mass of electron
e = 4.80320427e-10 # electron charge
sT = 6.65e-25 # thompson cross section
sqrt2 = np.sqrt(2) # square root of 2; probably doesn't save a lot time

class Lightcurve(object):
    def __init__(self,
                 ### physical parameters ###
                 nu_obs = 7e14,             # observation frequency
                 E_iso = 1e48,              # fireball isotropic energy
                 G_0 = 100,                 # the initial Lorentz factor of the fireball
                 theta_j = 90,              # jet opening angle, degrees
                 theta_obs = 0,             # observer angle, degrees
                 n_ism = 0.1,		        # the number density of the interstellar medium
                 ee = 1e-2, 		        # electron equipartition parameter
                 eB = 0.005,			    # B field equipartition parameter
                 pel = 2.5, 				# electron acceleration slope
                 jet_type = "homogenous",   # homogenous, structured, gaussian, or numerical
                 a_e = 2,                   # the parameters which control the shape of the structured jet.  Refer to Rossi+ 2002.
                 b_e = 1,
                 a_G = 1,
                 b_G = 1,
                 theta_c,                   # core angular size, degrees- refer to Rossi.
                 
                 
                 ### simulation parameters ###
                 dt = .1,                   # timestep size, in seconds
                 radius_range = (5,20),     # log of the min and max simulated radius, cm
                 dr = .001,                 # stepsize of the radius
                 n_theta = 500,             # number of theta bins
                 n_phi = 10):               # number of phi bins
        
        ### physical parameters ###
        self.nu_obs = nu_obs
        self.E_iso = E_iso
        self.G_0 = G_0
        self.n_ism = n_ism		           # the number density of the interstellar medium
        self.ee = ee			           # electron equipartition parameter
        self.eB = eB				       # B field equipartition parameter
        self.pel = pel 					   # electron acceleration slope
        self.theta_j = theta_j * np.pi / 180
        self.theta_obs = theta_obs * np.pi / 180
        self.jet_type = jet_type
        self.theta_c = theta_c * np.pi/180
        
        ### simulation parameters ###
        self.n_theta = n_theta
        self.d_theta = self.theta_j / n_theta		
        self.n_phi = n_phi		
        self.d_phi = 2 * np.pi / n_phi	
        self.dt = dt                
        self.start_radius = radius_range[0]
        self.end_radius = radius_range[1]
        self.dr = dr
    


    # same usage as np.arange but returns a logarithmic array like np.logspace
    def logrange(self, start, stop, dx):
        return(10**np.arange(start, stop, dx))
    
    def get_t_obs(self, start_time, end_time):
        times = self.logrange(start_time,
                              end_time,
                              self.dt)
        times *= 86400   # convert to seconds
        return(times)
        
    def get_radii(self):
        radii = self.logrange(self.start_radius,
                              self.end_radius,
                              self.dr)
        return(radii)
    
    def update_f(self, r):
        f = self.G_0 * cc**2 / self.E_iso * 4 * np.pi / 3. * self.n_ism * m_p * r**3
        return(f)
    
    def update_G(self, f):
        numerator = 1 + (4 * self.G_0 * f + 4 * f**2)
        denominator = 2 * f
        G = (np.sqrt(numerator) - 1) / denominator
        
        ### check for if it is bigger than G_0, and change everything up until then
        greater_than_G_0 = np.where(G > self.G_0)[0]
        if greater_than_G_0.size > 0:
            G[:greater_than_G_0[-1]] = self.G_0
        
        return(G)
    
    def update_G_sh(self, G):
        return(1 + sqrt2 * (G - 1))
    
    def update_beta_sh(self, G_sh):
        return(np.sqrt(1 - (1 / (G_sh**2))))
    
    def get_t_lab(self, length, r, b_sh):
        t_lab = np.zeros(length)
        for i in xrange(length):
            t_lab[i] = np.trapz(1 / (cc * b_sh[:i+1]),
                                x = r[:i+1])
        
        t_lab += r[0] / (b_sh[0] * cc)
        
        return(t_lab)
    
    def get_energy_per_sa(self, thetas):
        energy_per_sa = np.zeros_like(thetas)
        if self.jet_type == "homogenous":
            energy_per_sa += self.E_iso
        
        elif self.jet_type = "structured":
            denominator = (1 + (thetas / self.theta_c)**(self.a_e * self.b_e))
            energy_per_sa += self.E_iso / (denominator**(1/b_e))
        
        elif self.jet_type = "gaussian":
            energy_per_sa += self.E_iso * np.exp(-((thetas**2) / (2 * self.theta_c**2)))
    
        return(energy_per_sa)

    def get_lorentz_per_sa(self, thetas):
        lorentz_per_sa = np.zeros_like(thetas)
        if self.jet_type == "homogenous" or self.jet_type = "gaussian":
            lorentz_per_sa += self.G_0
            
        elif self.jet_type == "structured":
            denominator = (1 + (thetas / self.theta_c)**(self.a_G * self.b_G))
            energy_per_sa += self.G_0 / (denominator**(1/b_G))
    
        return(lorentz_per_sa)
        
    def time_evolve(self, start_time = -2.5, end_time = 2.1):
        if start_time > end_time:
            raise ValueError("Make sure the start time is before the end time.  In this case, {} is after {}.".format(start_time, end_time))
            
        if start_time is None:
            start_time = t_obs[-1]
        
        ### establish all of the quantities for the duration of the integration ###
        ### these should all be in one recarray accessible to the whole object
        rad = self.get_radii()
        num_of_radii = len(rad)
        f = self.update_f(rad)
        G = self.update_G(f)
        G_sh = self.update_G_sh(G)
        beta_sh = self.update_beta_sh(G_sh)
        t_lab = self.get_t_lab(num_of_radii, rad, beta_sh)
        t_obs = self.get_t_obs(start_time, end_time)
        
        ### this will store the output lightcurve data
        lightcurve = np.zeros((len(t_obs),2))
        lightcurve[:, 0] = t_obs
        
        theta_vals = np.arange(self.d_theta/2,
                               self.theta_j,
                               self.d_theta)
                               
        phi_vals = np.arange(self.d_phi/2,
                             2 * np.pi,
                             self.d_phi)
        
        vec_obs = np.array([np.sin(self.theta_obs),
                            0,
                            np.cos(self.theta_obs)])
        
        
        energy_dist = self.get_energy_per_sa(theta_vals)
        lorentz_dist = self.get_lorentz_per_sa(theta_vals)
        
        
        ### creating a bunch of arrays to hold various quantities on the EATS
        ### these should all be in one recarray like above
        the_theta=np.zeros([self.n_theta, self.n_phi])
        R_EATS=np.zeros([self.n_theta, self.n_phi])
        G_EATS=np.zeros([self.n_theta, self.n_phi])
        Iprime_nu=np.zeros([self.n_theta, self.n_phi])
        dL_nu=np.zeros([self.n_theta, self.n_phi])
        fs = np.zeros(len(t_obs))
        thetas2D=np.zeros([self.n_theta, self.n_phi])
 
        for i in xrange(self.n_phi):
            thetas2D[:, i] = theta_vals
        
        for timestep in xrange(len(t_obs)):
            for i_theta in xrange(self.n_theta):
                for i_phi in xrange(self.n_phi):
                    vec_axis = np.array([np.sin(theta_vals[i_theta]) * np.cos(phi_vals[i_phi]),
                                         np.sin(theta_vals[i_theta]) * np.sin(phi_vals[i_phi]),
                                         np.cos(theta_vals[i_theta])])
                    
                    the_theta[i_theta, i_phi] = np.arccos(np.dot(vec_axis, vec_obs))
                    
                    R_EATS[i_theta, i_phi] = np.interp(t_obs[timestep],
                                                       t_lab - (rad / cc) * np.cos(the_theta[i_theta, i_phi]),
                                                       rad)
                    
                    G_EATS[i_theta, i_phi] = np.interp(R_EATS[i_theta, i_phi], rad, G)
            feats = self.G_0 + np.sqrt(self.G_0**2 + G_EATS**2 - 1) / (2 * (G_EATS**2 - 1))
            gamma_inj = m_p / m_e * self.ee * (G_EATS - 1) + 1
            B = np.sqrt(32 * np.pi * self.eB * m_p * cc**2 * self.n_ism * (G_EATS**2 - 1))
            nu_inj = 0.25 * e * gamma_inj**2 * B / m_e / cc
            gamma_cool=15 * np.pi * m_e * cc**2 * np.sqrt(G_EATS**2 - 1) / sT / B**2 / R_EATS
            nu_cool = 0.25 * e * gamma_cool**2 * B / m_e / cc
            Pprime = 4. / 3. * sT * cc * B**2 / 8 / np.pi * (gamma_inj**2 - 1)
            B_EATS = np.sqrt(1. - 1. / G_EATS**2)
            delta = 1. / G_EATS / (1. - B_EATS * np.cos(the_theta))
            nu_prime = self.nu_obs / delta

            jj=np.where(nu_cool>nu_inj)
            if jj[0].size>0:
                nu_max=nu_inj
                Iprime_peak=Pprime*self.n_ism*R_EATS/nu_max
                kk=np.where(nu_prime<nu_inj)
                if kk[0].size>0:
                    Iprime_nu[kk]=Iprime_peak[kk]*(nu_prime[kk]/nu_inj[kk])**(1./3.)
                kk=np.where((nu_prime>=nu_inj)&(nu_prime<nu_cool))
                if kk[0].size>0:
                    Iprime_nu[kk]=Iprime_peak[kk]*(nu_prime[kk]/nu_inj[kk])**(-(self.pel-1)/2.)
                kk=np.where(nu_prime>=nu_cool)
                if kk[0].size>0:
                    Iprime_nu[kk]=Iprime_peak[kk]*(nu_inj[kk]/nu_cool[kk])**((self.pel-1)/2.)*(nu_prime[kk]/nu_cool[kk])**(-self.pel/2.)
            dL_nu[jj]=Iprime_nu[jj]*delta[jj]**3*R_EATS[jj]**2*np.sin(thetas2D[jj])*self.d_theta*self.d_phi
            jjo=jj

            jj=np.where(nu_cool<=nu_inj)
            if jj[0].size>0:
                nu_max=nu_cool
                Iprime_peak=Pprime*self.n_ism*R_EATS/nu_max
                kk=np.where(nu_prime<=nu_cool)
                if kk[0].size>0:
                    Iprime_nu[kk]=Iprime_peak[kk]*(nu_prime[kk]/nu_cool[kk])**(1./3.)
                kk=np.where((nu_prime<=nu_inj)&(nu_prime>nu_cool))
                if kk[0].size>0:
                    Iprime_nu[kk]=Iprime_peak[kk]*(nu_prime[kk]/nu_cool[kk])**(-1/2.)
                kk=np.where(nu_prime>nu_inj)
                if kk[0].size>0:
                    Iprime_nu[kk]=Iprime_peak[kk]*(nu_cool[kk]/nu_inj[kk])**(1./2.)*(nu_prime[kk]/nu_inj[kk])**(-self.pel/2.)
            dL_nu[jj]=Iprime_nu[jj]*delta[jj]**3*R_EATS[jj]**2*np.sin(thetas2D[jj])*self.d_theta*self.d_phi
            
            lightcurve[timestep, 1] = dL_nu.sum() 
            #print(dL_nu.sum())
            prog_str = '\r timestep {} of {}.'
            sys.stdout.write(prog_str.format(timestep, len(t_obs)))
            sys.stdout.flush()
            
        return(lightcurve)

    def make_curve(self, data, ax = None, **kwargs):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        lightcurve_plot = ax.loglog(data[:, 0], data[:, 1], **kwargs)
        return(lightcurve_plot)
    
    
    