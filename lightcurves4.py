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
from scipy import interpolate
import matplotlib.pyplot as plt
import sys

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
                 lorentz_distribution = None): # same as above, can specify "constant"
        
        ### physical parameters ###
        self.nu_obs = nu_obs
        self.E_iso = E_iso
        self.G_0 = G_0
        self.n_ism = n_ism		           # the number density of the interstellar medium
        self.ee = ee			           # electron equipartition parameter
        self.eB = eB				       # B field equipartition parameter
        self.pel = pel 					   # electron acceleration slope
        
        if energy_distribution is not None:
            self.theta_j = self.get_theta_j_from_data() # if the energy distribution is being read from a file, then you need to read the jet opening angle from the file, which I'll do later in the code
        else:
            self.theta_j = theta_j * np.pi / 180
        self.theta_obs = theta_obs * np.pi / 180
        self.jet_type = jet_type
        self.theta_c = theta_c * np.pi/180
        self.a_e = a_e
        self.b_e = b_e
        self.a_G = a_G
        self.b_G = b_G
        self.energy_distribution = energy_distribution
        self.lorentz_distribution = lorentz_distribution
        
        ### simulation parameters ###
        self.n_theta = n_theta
        self.d_theta = self.theta_j / n_theta		
        self.n_phi = n_phi		
        self.d_phi = 2 * np.pi / n_phi	
        self.dt = dt                
        self.start_radius = radius_range[0]
        self.end_radius = radius_range[1]
        self.dr = dr
        
        ### set the radius bins. this and theta_vals will be accessible to the whole object###
        self.r_vals = self.get_r_vals()
        self.th_vals = self.get_theta_vals()
        self.ph_vals = self.get_ph_vals()
        
    def get_theta_j(self):
        return(theta_j)
    
    def set_theta_j(self, new_theta_j):
        self.theta_j = new_theta_j
        return(self.theta_j)

    # same usage as np.arange but returns a logarithmic array like np.logspace
    def logrange(self, start, stop, dx):
        return(10**np.arange(start, stop, dx))
    
    def get_t_obs(self, start_time, end_time):
        times = self.logrange(start_time,
                              end_time,
                              self.dt)
        times *= 86400   # convert to seconds
        return(times)
    
    
    def get_theta_vals(self):
        theta_vals = np.arange(self.d_theta/2,
                               self.theta_j,
                               self.d_theta)
        return(theta_vals)
    
    def get_ph_vals(self):
        theta_vals = np.arange(self.d_phi/2,
                               2 * np.pi,
                               self.d_phi)
        return(phi_vals)
    
    ############################        
    #### constant functions ####
    ############################
    def generate_energy_per_sa(self, thetas):
        energy_per_sa = np.zeros_like(thetas)
        if self.jet_type == "homogenous":
            energy_per_sa += self.E_iso
        
        elif self.jet_type == "structured":
            denominator = (1 + (thetas / self.theta_c)**(self.a_e * self.b_e))
            energy_per_sa += self.E_iso / (denominator**(1 / self.b_e))
        
        elif self.jet_type == "gaussian":
            energy_per_sa += self.E_iso * np.exp(-((thetas**2) / (2 * self.theta_c**2)))
        
        elif self.jet_type == "numerical":
            energy_func = interpolate.interp1d(self.energy_distribution[:, 0],
                                               self.energy_distribution[:, 1])
            
            energy_per_sa = energy_func(thetas)
            
        return(energy_per_sa)

    def generate_lorentz_per_sa(self, thetas):
        lorentz_per_sa = np.zeros_like(thetas)
        if self.jet_type == "homogenous" or self.jet_type == "gaussian" or self.lorentz_distribution == "constant":
            lorentz_per_sa += self.G_0
            
        elif self.jet_type == "structured":
            denominator = (1 + (thetas / self.theta_c)**(self.a_G * self.b_G))
            lorentz_per_sa += self.G_0 / (denominator**(1 / self.b_G))
        
        elif self.jet_type == "numerical" and self.lorentz_distribution is not "constant":
            lorentz_func = interpolate.interp1d(self.lorentz_distribution[:, 0],
                                                self.lorentz_distribution[:, 1])
            lorentz_per_sa = lorentz_func(thetas)
        
        return(lorentz_per_sa)
    
    def get_radii(self):
        radii = self.logrange(self.start_radius,
                              self.end_radius,
                              self.dr)
        return(radii)
    
    def get_f(self, r, energy_dist, G_dist):
        f = G_dist * cc**2 / energy_dist * 4 * np.pi / 3. * self.n_ism * m_p * r**3
        return(f)
    
    def get_G(self, f, G_dist):
        numerator = 1 + (4 * G_dist * f + 4 * f**2)
        denominator = 2 * f
        G = (np.sqrt(numerator) - 1) / denominator
        
        ### check for if it is bigger than G_0, and change everything up until then
        greater_than_G_0 = np.where(G > self.G_0)[0]
        if greater_than_G_0.size > 0:
            G[:greater_than_G_0[-1]] = self.G_0
        
        return(G)
    
    def get_G_sh(self, G):
        return(1 + sqrt2 * (G - 1))
    
    def get_beta_sh(self, G_sh):
        return(np.sqrt(1 - (1 / (G_sh**2))))
    
    def time_of_flight_delay(self, t_lab, rad, angle):
        return(t_lab - (rad / cc) * np.cos(angle))

    
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
        
        print('')
        return(t_lab)        
    
    # this returns a dictionary instead of an array because it's not super memory intensive
    # and it holds both the radius arrays (which are 1 x n_rad) and the meshes (which are
    # n_rad x n_theta)
    def make_time_independent_dict(self):

        r_mesh, th_mesh = np.meshgrid(self.r_vals,
                                      self.theta_vals,
                                      indexing = 'ij')

        
        ti_arr_dtypes = [('r_mesh', 'float'),
                         ('th_mesh', 'float'),
                         ('energy_dist', 'float'),
                         ('lorentz_dist', 'float'),
                         ('f', 'float'),
                         ('G', 'float'),
                         ('G_sh', 'float'),
                         ('beta_sh', 'float'),
                         ('t_lab', 'float')]
        
        ti_arr = np.zeros_like(r_mesh, dtype = ti_arr_dtypes)
        
        ti_arr['r_mesh'] = r_mesh
        
        ti_arr['th_mesh'] = th_mesh
        
        ti_arr['energy_dist'] = self.get_energy_per_sa(self.theta_vals)
        
        ti_arr['lorentz_dist'] = self.get_lorentz_per_sa(self.theta_vals)
        
        ti_arr['f'] = self.get_f(r_mesh,
                                 ti_arr['energy_dist'],
                                 ti_arr['lorentz_dist'])
                                          
        ti_arr['G'] = self.get_G(ti_arr['f'],
                                 ti_arr['lorentz_dist'])
                                          
        ti_arr['G_sh'] = self.get_G_sh(ti_arr['G'])
        
        ti_arr['beta_sh'] = self.get_beta_sh(ti_arr['G_sh'])
        
        ti_arr['t_lab'] = self.get_t_lab(self.r_vals,
                                         ti_arr['beta_sh'])
        
        return(ti_arr)
    
    def new_lightcurve(self, t_obs):
        lightcurve = np.zeros((len(t_obs),2))
        lightcurve[:, 0] = t_obs
    
        return(lightcurve)
    ###############
    
    
    
    ############################
    #### variable functions ####
    ############################
    def update_gamma_inj(self, g_eats):
        return(m_p / m_e * self.ee * (g_eats - 1) + 1)
    
    def update_B(self, g_eats):
        return(np.sqrt(32 * np.pi * self.eB * m_p * cc**2 * self.n_ism * (g_eats**2 - 1)))

    def update_nu_inj(self, gamma_inj):
        return((0.25 * e * gamma_inj**2 * B) / (m_e * cc))
    
    def update_gamma_cool(self, g_eats, r_eats, B):
        return(15 * np.pi * m_e * cc**2 * np.sqrt(g_eats**2 - 1) / (sT * B**2 * r_eats))
        
    def update_nu_cool(self, gamma_cool):
        retrun(0.25 * e * gamma_cool**2 * B / (m_e * cc))
    
    def update_Pprime(self, B, gamma_inj):
         return((4 / 3) * sT * cc * B**2 / (8 * np.pi) * (gamma_inj**2 - 1))
    
    def update_B_EATS(self, g_eats):
        return(np.sqrt(1 - (1 / g_eats**2)))
    
    def update delta(self, g_eats, b_eats, eff_theta)
        return(1 / (g_eats * (1 - b_eats * np.cos(eff_theta))))
    
    def update_nu_prime(self, delta):
        return(self.nu_obs / delta)
    
    def update_Iprime_nu(self, r_eats, nu_cool, nu_inj, nu_prime):      
        Iprime_nu = np.zeros_like(r_eats)
        
        nu_cool_greater = np.where(nu_cool > nu_inj)
        if nu_cool_greater[0].size>0:
            nu_max = nu_inj
            Iprime_peak = Pprime * self.n_ism * r_eats / nu_max
            
            nu_p_less = np.where(nu_prime < nu_inj)
            if nu_p_less[0].size > 0:
                Iprime_nu[nu_p_less] = Iprime_peak[nu_p_less] * (nu_prime[nu_p_less] / nu_inj[nu_p_less])**(1 / 3)
            
            nu_p_mid=np.where((nu_prime >= nu_inj) & (nu_prime < nu_cool))
            if nu_p_mid[0].size > 0:
                Iprime_nu[nu_p_mid] = Iprime_peak[nu_p_mid] * (nu_prime[nu_p_mid] / nu_inj[nu_p_mid])**(-(self.pel-1) / 2)
            
            nu_p_greater=np.where(nu_prime >= nu_cool)
            if nu_p_greater[0].size > 0:
                Iprime_nu[nu_p_greater] = Iprime_peak[nu_p_greater] * (nu_inj[nu_p_greater] / nu_cool[nu_p_greater])**((self.pel-1) / 2) * (nu_prime[nu_p_greater] / nu_cool[nu_p_greater])**(-self.pel / 2)
        

        nu_cool_less = np.where(nu_cool <= nu_inj)
        if nu_cool_less[0].size > 0:
            nu_max = nu_cool
            Iprime_peak = Pprime * self.n_ism * r_eats / nu_max
            
            nu_p_less = np.where(nu_prime <= nu_cool)
            if nu_p_less[0].size > 0:
                Iprime_nu[nu_p_less] = Iprime_peak[nu_p_less] * (nu_prime[nu_p_less] / nu_cool[nu_p_less])**(1 / 3)
                
            nu_p_mid = np.where((nu_prime <= nu_inj) & (nu_prime > nu_cool))
            if nu_p_mid[0].size > 0:
                Iprime_nu[nu_p_mid] = Iprime_peak[nu_p_mid] * (nu_prime[nu_p_mid] / nu_cool[nu_p_mid])**(-1 / 2)
                
            nu_p_greater=np.where(nu_prime > nu_inj)
            if nu_p_greater[0].size > 0:
                Iprime_nu[nu_p_greater] = Iprime_peak[nu_p_greater] * (nu_cool[nu_p_greater] / nu_inj[nu_p_greater])**(1 / 2) * (nu_prime[nu_p_greater] / nu_inj[nu_p_greater])**(-self.pel / 2)
            
        return(Iprime_nu)
    
    def update_dL_nu(self, r_eats, Iprime_nu, thetas2D):
        dL_nu = np.zeros_like(Iprime_nu)
        
        nu_cool_greater = np.where(nu_cool > nu_inj)
        if nu_cool_greater[0].size > 0:  
            dL_nu = Iprime_nu[nu_cool_greater] * delta[nu_cool_greater]**3 * r_eats[nu_cool_greater]**2 * np.sin(thetas2D[nu_cool_greater]) * self.d_theta * self.d_phi
        
        nu_cool_less = np.where(nu_cool <= nu_inj)
        if nu_cool_less[0].size > 0:
            dL_nu = Iprime_nu[nu_cool_less] * delta[nu_cool_less]**3 * r_eats[nu_cool_less]**2 * np.sin(thetas2D[nu_cool_less]) * self.d_theta * self.d_phi

    return(dL_nu)
            
    
    def make_EATS_arr(self):
        EATS_dtypes = [('R_EATS', 'float'),
                       ('G_EATS', 'float'),
                       ('eff_theta', 'float'),
                       ('thetas2D', 'float'),
                       ('gamma_inj', 'float'),
                       ('B', 'float'),
                       ('nu_inj', 'float'),
                       ('gamma_cool', 'float'),
                       ('nu_cool', 'float'),
                       ('Pprime', 'float'),
                       ('B_EATS', 'float'),
                       ('delta', 'float'),
                       ('nu_prime', 'float'),
                       ('Iprime_nu', 'float'),
                       ('dL_nu', 'float')]
                       
        EATS_array = np.zeros([self.n_theta, self.n_phi], dtype = EATS_dtypes)
        return()
    
    def update_EATS_arr(self, EATS_arr):    
        EATS_arr['gamma_inj'] = self.update_gamma_inj(EATS_arr['G_EATS'])
        
        EATS_arr['B'] = self.update_B(EATS_arr['G_EATS'])
        
        EATS_arr['nu_inj'] = self.update_nu_inj(EATS_arr['gamma_inj'])
        
        EATS_arr['gamma_cool'] = self.update_gamma_cool(EATS_arr['G_EATS'],
                                                        EATS_arr['R_EATS'],
                                                        EATS_arr['B'])
        
        EATS_arr['nu_cool'] = self.update_nu_cool(EATS_arr['gamma_cool'])
        
        EATS_arr['Pprime'] = self.update_Pprime(EATS_arr['B'], EATS_arr['gamma_inj'])
        
        EATS_arr['B_EATS'] = self.update_B_EATS(EATS_arr['G_EATS'])
        
        EATS_arr['delta'] = self.update_delta(EATS_arr['G_EATS'],
                                              EATS_arr['B_EATS'],
                                              EATS_arr['eff_theta'])
        
        EATS_arr['nu_prime'] = self.update_nu_prime(EATS_arr['delta'])
        
        EATS_arr['Iprime_nu'] = self.update_Iprime_nu(EATS_arr['R_EATS'],
                                                      EATS_arr['nu_cool'],
                                                      EATS_arr['nu_inj'],
                                                      EATS_arr['nu_prime'])
        
        EATS_arr['dL_nu'] = self.update_dL_nu(EATS_arr['Iprime_nu', EATS_arr['thetas2D']])
        return(EATS_arr)
    ###############
    
    ##################################
    #### time-evolve the fireball ####
    ##################################
    def time_evolve(self, start_time = -2.5, end_time = 2.1):
        if start_time > end_time:
            raise ValueError("Make sure the start time is before the end time.  In this case, {} is after {}.".format(start_time, end_time))
            
        if start_time is None:
            start_time = t_obs[-1]
        
        ### establish all of the quantities for the duration of the integration
        time_ind_arr = self.make_time_independent_dict()
        
        ### t_obs is the clock for the lightcurve data
        t_obs = self.get_t_obs(start_time, end_time)
        
        ### this will store the output lightcurve data
        lightcurve = self.new_lightcurve(t_obs)
        
        vec_obs = np.array([np.sin(self.theta_obs),
                            0,
                            np.cos(self.theta_obs)])
        
        # update all the quantities like gamma_inj, B, nu_inj etc...
        EATS_arr = self.make_EATS_arr()
        EATS_arr['thetas2D'] = np.meshgrid(time_ind_arr['Th'], time_ind_arr['Ph'])[0]
        
        for timestep in xrange(len(t_obs)):
            for i_theta in xrange(self.n_theta):
                for i_phi in xrange(self.n_phi):
                    
                    ### These lines calculate the surface as seen by an observer###
                    
                    vec_axis = np.array([np.sin(self.th_vals[i_theta]) * np.cos(self.ph_vals[i_phi]),
                                         np.sin(self.th_vals[i_theta]) * np.sin(self.ph_vals[i_phi]),
                                         np.cos(self.th_vals[i_theta])])
                    
                    EATS_arr['eff_theta'][i_theta, i_phi] = np.arccos(np.dot(vec_axis, vec_obs))
                    
                    time_of_flight = self.time_of_flight_delay(time_ind_arr['t_lab'][:, i_theta],
                                                               self.r_vals,
                                                               eff_theta[i_theta, i_phi])
                    
                    EATS_arr['R_EATS'][i_theta, i_phi] = np.interp(t_obs[timestep],
                                                                   time_of_flight,
                                                                   self.r_vals)
                    
                    EATS_arr['G_EATS'][i_theta, i_phi] = np.interp(EATS_arr['R_EATS'][i_theta, i_phi],
                                                                   self.r_vals,
                                                                   time_ind_arr['G'][:, i_theta])


            
            ### Calculate all the quantities associated with the new surface
            EATS_arr = self.update_EATS_vals(EATS_arr)
            
            ### Add the luminosity to the lightcurve at this timestep
            lightcurve[timestep, 1] = EATS_arr['dL_nu'].sum() 
            
            ### Update the screen with the progress
            prog_str = '\r Time evolving... {:.1F}%.'
            sys.stdout.write(prog_str.format(timestep / len(t_obs) * 100))
            sys.stdout.flush()
            
        return(lightcurve)
        
    ######################
    #### make a plot! ####
    ######################
    def make_curve(self, data, ax = None, **kwargs):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        lightcurve_plot = ax.loglog(data[:, 0], data[:, 1], **kwargs)
        return(lightcurve_plot)
    
    
    