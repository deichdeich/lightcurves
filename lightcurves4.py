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
"""
yes, everything is a function of r and theta.  iterate through the possible values for
every thing that depends on energy and lorentz, including t_lab and R_EATS

make sure you can put in arbitrary arrays of energy-per-sa and lorentz-per-sa
"""

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
                 theta_c = 90,                   # core angular size, degrees- refer to Rossi.
                 
                 
                 ### simulation parameters ###
                 dt = .1,                   # timestep size, in seconds
                 radius_range = (5,20),     # log of the min and max simulated radius, cm
                 dr = .001,                 # stepsize of the radius
                 n_theta = 500,             # number of theta bins
                 n_phi = 10,                # number of phi bins
                 energy_distribution = None, # file path for a numerical distribution
                 lorentz_distribution = None): # same as above
        
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
        
        ### time independent quantity column names ###
        self.ti_colnames = [('r', 'float'),
                            ('Th', 'float'),
                            ('Ph', 'float'),
                            ('f', 'float'),
                            ('G', 'float'),
                            ('G_sh', 'float'),
                            ('beta_sh', 'float'),
                            ('t_lab', 'float')]
        
    


    # same usage as np.arange but returns a logarithmic array like np.logspace
    def logrange(self, start, stop, dx):
        return(10**np.arange(start, stop, dx))
    
    def get_t_obs(self, start_time, end_time):
        times = self.logrange(start_time,
                              end_time,
                              self.dt)
        times *= 86400   # convert to seconds
        return(times)
    
    def get_energy_per_sa(self, thetas):
        energy_per_sa = np.zeros_like(thetas)
        if self.jet_type == "homogenous":
            energy_per_sa += self.E_iso
        
        elif self.jet_type == "structured":
            denominator = (1 + (thetas / self.theta_c)**(self.a_e * self.b_e))
            energy_per_sa += self.E_iso / (denominator**(1/b_e))
        
        elif self.jet_type == "gaussian":
            energy_per_sa += self.E_iso * np.exp(-((thetas**2) / (2 * self.theta_c**2)))
    
        return(energy_per_sa)

    def get_lorentz_per_sa(self, thetas):
        lorentz_per_sa = np.zeros_like(thetas)
        if self.jet_type == "homogenous" or self.jet_type == "gaussian":
            lorentz_per_sa += self.G_0
            
        elif self.jet_type == "structured":
            denominator = (1 + (thetas / self.theta_c)**(self.a_G * self.b_G))
            lorentz_per_sa += self.G_0 / (denominator**(1/b_G))
        
        return(lorentz_per_sa)
    
    def get_radii(self):
        radii = self.logrange(self.start_radius,
                              self.end_radius,
                              self.dr)
        return(radii)
    
    def update_f(self, r, energy_dist, G_dist):
        f = G_dist * cc**2 / energy_dist * 4 * np.pi / 3. * self.n_ism * m_p * r**3
        return(f)
    
    def update_G(self, f, G_dist):
        numerator = 1 + (4 * G_dist * f + 4 * f**2)
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
    
    def get_t_lab(self, r, b_sh):
    
        r_length, th_length = b_sh.shape
        
        t_lab = np.zeros_like(b_sh)


        for i_th in xrange(th_length):
            prog_str = '\r Calculating laboratory time... {:.1F}%'
            sys.stdout.write(prog_str.format((i_th / th_length) * 100))
            sys.stdout.flush()
            for i_r in xrange(r_length):
                t_lab[i_r][i_th] = np.trapz(1 / (cc * b_sh[:i_r+1, i_th]),
                                            x = r[:i_r+1])
        
            t_lab[:, i_th] += r[0] / (b_sh[0][i_th] * cc)
        
        return(t_lab)
    
    def time_of_flight_delay(self, t_lab, rad, angle):
        return(t_lab - (rad / cc) * np.cos(angle))
        
    
    def make_time_independent_array(self):
        rad = self.get_radii()
        theta_vals = np.arange(self.d_theta/2,
                               self.theta_j,
                               self.d_theta)
                               
        phi_vals = np.arange(self.d_phi/2,
                             2 * np.pi,
                             self.d_phi)
        

        r_mesh, th_mesh = np.meshgrid(rad, theta_vals, indexing = 'ij')

        ti_dict = {}
        
        ti_dict['r'] = rad
        
        ti_dict['Th'] = theta_vals
        
        ti_dict['Ph'] = phi_vals
        
        ti_dict['r_mesh'] = r_mesh
        
        ti_dict['th_mesh'] = th_mesh
        
        ti_dict['energy_dist'] = self.get_energy_per_sa(theta_vals)
        
        ti_dict['lorentz_dist'] = self.get_lorentz_per_sa(theta_vals)
        
        ti_dict['f_mesh'] = self.update_f(r_mesh,
                                          ti_dict['energy_dist'],
                                          ti_dict['lorentz_dist'])
                                          
        ti_dict['G_mesh'] = self.update_G(ti_dict['f_mesh'],
                                          ti_dict['lorentz_dist'])
                                          
        ti_dict['G_sh_mesh'] = self.update_G_sh(ti_dict['G_mesh'])
        
        ti_dict['beta_sh_mesh'] = self.update_beta_sh(ti_dict['G_sh_mesh'])
        
        ti_dict['t_lab_mesh'] = self.get_t_lab(ti_dict['r'], ti_dict['beta_sh_mesh'])
        
        return(ti_dict)
    

    def time_evolve(self, start_time = -2.5, end_time = 2.1):
        if start_time > end_time:
            raise ValueError("Make sure the start time is before the end time.  In this case, {} is after {}.".format(start_time, end_time))
            
        if start_time is None:
            start_time = t_obs[-1]
        
        ### establish all of the quantities for the duration of the integration
        time_ind_arr = self.make_time_independent_array()
        
        ### t_obs is the clock for the lightcurve data
        t_obs = self.get_t_obs(start_time, end_time)
        
        ### this will store the output lightcurve data
        lightcurve = np.zeros((len(t_obs),2))
        lightcurve[:, 0] = t_obs
        
        vec_obs = np.array([np.sin(self.theta_obs),
                            0,
                            np.cos(self.theta_obs)])
        
        energy_dist = self.get_energy_per_sa(theta_vals)
        lorentz_dist = self.get_lorentz_per_sa(theta_vals)
        
        ### creating a bunch of arrays to hold various quantities on the EATS
        ### these should all be in one recarray like above
        eff_theta = np.zeros([self.n_theta, self.n_phi]) # eff_theta is the effective angle created by the observer angle and the geometry of the EATS
        R_EATS = np.zeros([self.n_theta, self.n_phi])
        G_EATS = np.zeros([self.n_theta, self.n_phi])
        Iprime_nu = np.zeros([self.n_theta, self.n_phi])
        dL_nu = np.zeros([self.n_theta, self.n_phi])
        thetas2D = np.zeros([self.n_theta, self.n_phi])
 
        for i in xrange(self.n_phi):
            thetas2D[:, i] = theta_vals
        
        for timestep in xrange(len(t_obs)):
            for i_theta in xrange(self.n_theta):
                for i_phi in xrange(self.n_phi):
                    
                    vec_axis = np.array([np.sin(theta_vals[i_theta]) * np.cos(phi_vals[i_phi]),
                                         np.sin(theta_vals[i_theta]) * np.sin(phi_vals[i_phi]),
                                         np.cos(theta_vals[i_theta])])
                    
                    eff_theta[i_theta, i_phi] = np.arccos(np.dot(vec_axis, vec_obs))
                    
                    time_of_flight = self.time_of_flight_delay(time_ind_arr['t_lab'],
                                                               time_ind_arr['r'],
                                                               eff_theta[i_theta, i_phi])
                    
                    R_EATS[i_theta, i_phi] = np.interp(t_obs[timestep],
                                                       time_of_flight,
                                                       time_ind_arr['r'])
                    
                    G_EATS[i_theta, i_phi] = np.interp(R_EATS[i_theta, i_phi],
                                                       time_ind_arr['r'],
                                                       time_ind_arr['G'])

            gamma_inj = m_p / m_e * self.ee * (G_EATS - 1) + 1
            B = np.sqrt(32 * np.pi * self.eB * m_p * cc**2 * self.n_ism * (G_EATS**2 - 1))
            nu_inj = 0.25 * e * gamma_inj**2 * B / m_e / cc
            gamma_cool=15 * np.pi * m_e * cc**2 * np.sqrt(G_EATS**2 - 1) / sT / B**2 / R_EATS
            nu_cool = 0.25 * e * gamma_cool**2 * B / m_e / cc
            Pprime = 4. / 3. * sT * cc * B**2 / 8 / np.pi * (gamma_inj**2 - 1)
            B_EATS = np.sqrt(1. - 1. / G_EATS**2)
            delta = 1. / G_EATS / (1. - B_EATS * np.cos(eff_theta))
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
    
    
    