"""
grblc.py: GRB Lightcurve Calculator
Author: Alex Deich
E-mail: alex.d.deich@gmail.com
Github: github.com/deichdeich/lightcurves
Date: July, 2017

Full documentation available on the GitHub wiki.

All equation numbers refer to Rossi et al. (2002) unless otherwise noted.

Based on code by D. Lazzati at Oregon State University,
lazzatid@science.oregonstate.edu

This version currently fails for G_0 > 2739.
"""
from __future__ import division, print_function
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import sys

### Some global constants ###
cc = 2.997e10  # speed of light in cm/s
m_p = 1.67e-24 # mass of proton
m_e = m_p / 1836. # mass of electron
e = 4.80320427e-10 # electron charge
sT = 6.65e-25 # thompson cross section
sqrt2 = np.sqrt(2) # square root of 2; probably doesn't save a lot time

class Lightcurve(object):
    def __init__(self,
                 ### simulation parameters ###
                 n_theta,                      # number of theta bins
                 n_phi,                        # number of phi bins
                 dt = .001,                    # timestep size, in seconds
                 radius_range = (5,20),        # log of the min and max simulated radius, cm
                 dr = .001,                    # stepsize of the radius
                 energy_distribution = None,   # file path for a numerical distribution
                 lorentz_distribution = None, # same as above, can specify "constant"
                 
                 ### physical parameters ###
                 nu_obs = 7e14,             # observing frequency
                 E_iso = 1e48,              # fireball isotropic energy
                 G_0 = 100,                 # the initial Lorentz factor of the fireball
                 theta_j = 90,              # jet opening angle, degrees
                 theta_obs = 0,             # observer angle, degrees
                 n_ism = 0.1,               # the number density of the interstellar medium
                 ee = 1e-2,                 # electron equipartition parameter
                 eB = 0.005,                # B field equipartition parameter
                 pel = 2.5,                 # electron acceleration slope
                 jet_type = "homogenous",   # homogenous, structured, gaussian, or numerical
                 a_e = 2,                   # the parameters which control the shape of the structured jet.  Refer to Rossi+ 2002.
                 b_e = 1,
                 a_G = 1,
                 b_G = 1,
                 theta_c = 90,                   # core angular size, degrees- refer to Rossi.
                 z = 1.):                      # redshift
        
        ### physical parameters ###
        self.nu_obs = nu_obs               # observing frequency
        self.E_iso = E_iso                 # isotropic energy (for homogenous fireballs)
        self.G_0 = G_0                     # initial lorentz factor (for homogenous fireballs)
        self.n_ism = n_ism                   # the number density of the interstellar medium
        self.ee = ee                       # electron equipartition parameter
        self.eB = eB                       # B field equipartition parameter
        self.pel = pel                        # electron acceleration slope
        
        self.theta_obs = theta_obs * np.pi / 180
        self.jet_type = jet_type                 
        self.theta_c = theta_c * np.pi/180
        self.a_e = a_e
        self.b_e = b_e
        self.a_G = a_G
        self.b_G = b_G
        self.z = z
        
        
        if energy_distribution is not None:
            self.energy_distribution = np.genfromtxt(energy_distribution)
            self.theta_j = max(self.energy_distribution[:, 0]) # if the energy distribution is being read from a file, then you need to read the jet opening angle from the file
        else:
            self.theta_j = theta_j * np.pi / 180
        

        if lorentz_distribution == "constant":
            self.lorentz_distribution = lorentz_distribution
        elif lorentz_distribution is not "constant" and lorentz_distribution is not None:
            self.lorentz_distribution = np.genfromtxt(lorentz_distribution)
            # is the lorentz data binned the same as the energy data?
            self.theta_j_check(self.energy_distribution, self.lorentz_distribution)
        
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
    
    
    # can't extrapolate from the data, so you have to read the jet opening angle directly
    # from the energy and lorentz input data.  if they don't agree, it's bad.
    def theta_j_check(self, energy, lorentz):
        emax = max(energy[:, 0])
        lmax = max(lorentz[:, 0])
        if (emax - lmax) > 0.01:
            raise(ValueError('Your energy and lorentz distributons are not binned in the same way.  Cannot get reliable theta_j.  Energy theta_j: {}.  Lorentz theta_j: {}'.format(emax, lmax)))      
   
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
        ph_vals = np.arange(self.d_phi/2,
                               2 * np.pi,
                               self.d_phi)
        return(ph_vals)
    
    ############################        
    #### constant functions ####
    ############################
    # 'sa' = solid angle
    def generate_energy_per_sa(self, thetas):
        # homogenous jets have constant isotropic energy
        if self.jet_type == "homogenous":
            energy_per_sa = np.zeros_like(thetas) + self.E_iso
        
        elif self.jet_type == "structured":
            #  eq. 1
            denominator = (1 + (thetas / self.theta_c)**(self.a_e * self.b_e))
            energy_per_sa = self.E_iso / (denominator**(1 / self.b_e))
        
        elif self.jet_type == "gaussian":
            #  eq. 3
            energy_per_sa = self.E_iso * np.exp(-((thetas**2) / (2 * self.theta_c**2)))
        
        elif self.jet_type == "numerical":
            # this makes a custom function for the energy per solid angle,
            # which can be sampled at whatever binning you want, regardless the input data.
            energy_func = interpolate.interp1d(self.energy_distribution[:, 0],
                                               self.energy_distribution[:, 1])
            
            energy_per_sa = energy_func(thetas)
            
        return(energy_per_sa)

    def generate_lorentz_per_sa(self, thetas):
        if self.jet_type == "homogenous" or self.jet_type == "gaussian" or self.lorentz_distribution == "constant":
            lorentz_per_sa = np.zeros_like(thetas) + self.G_0
            
        elif self.jet_type == "structured":
            #  eq. 2
            denominator = (1 + (thetas / self.theta_c)**(self.a_G * self.b_G))
            lorentz_per_sa = self.G_0 / (denominator**(1 / self.b_G))
        
        elif self.jet_type == "numerical" and self.lorentz_distribution is not "constant":
            lorentz_func = interpolate.interp1d(self.lorentz_distribution[:, 0],
                                                self.lorentz_distribution[:, 1])
            lorentz_per_sa = lorentz_func(thetas)
        
        return(lorentz_per_sa)
    
    def get_r_vals(self):
        radii = self.logrange(self.start_radius,
                              self.end_radius,
                              self.dr)
        return(radii)
    
    def get_f(self, r, energy_dist, G_dist):
        #  eq. 5
        f = G_dist * cc**2 / energy_dist * 4 * np.pi / 3. * self.n_ism * m_p * r**3
        return(f)
    
    def get_G(self, f, G_dist):
        #  eq. 4
        numerator = 1 + (4 * G_dist * f + 4 * f**2)
        denominator = 2 * f
        G = (np.sqrt(numerator) - 1) / denominator
        
        ### check for if it is bigger than G_0, and change everything up until then
        greater_than_G_0 = np.where(G > self.G_0)[0]
        if greater_than_G_0.size > 0:
            G[:greater_than_G_0[-1]] = self.G_0
        
        return(G)
    
    def get_G_sh(self, G):
        #  eq.
        return(1 + sqrt2 * (G - 1))
    
    def get_beta_sh(self, G_sh):
        return(np.sqrt(1 - (1 / (G_sh**2))))
    
    def time_of_flight_delay(self, t_lab, rad, angle):
        return(t_lab - (rad / cc) * np.cos(angle))

    
    def get_t_lab(self, r, b_sh):
        '''
        Integrates along a line from the surface to the observer. Generates EATS.
        '''
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
    

    def make_time_independent_arr(self):
        '''
        The 'time independent array' is the array holding the values that don't change
        over the course of the integration, which are all computed prior.
        '''
        r_grid, th_grid = np.meshgrid(self.r_vals,
                                      self.th_vals,
                                      indexing = 'ij')

        
        ti_arr_dtypes = [('r_grid', 'float'),
                         ('th_grid', 'float'),
                         ('energy_dist', 'float'),
                         ('lorentz_dist', 'float'),
                         ('f', 'float'),
                         ('G', 'float'),
                         ('G_sh', 'float'),
                         ('beta_sh', 'float'),
                         ('t_lab', 'float')]
        
        ti_arr = np.zeros_like(r_grid, dtype = ti_arr_dtypes)
        
        ti_arr['r_grid'] = r_grid
        
        ti_arr['th_grid'] = th_grid
        
        ti_arr['energy_dist'] = self.generate_energy_per_sa(self.th_vals)
        
        ti_arr['lorentz_dist'] = self.generate_lorentz_per_sa(self.th_vals)
        
        ti_arr['f'] = self.get_f(r_grid,
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
        '''
        This just makes an empty array to hold the lightcurve.
        '''
        lightcurve = np.zeros((len(t_obs),2))
        lightcurve[:, 0] = t_obs
    
        return(lightcurve)
    ###############
    
    def beta_from_gamma(self, g_eats):
        return(np.sqrt(1 - (1 / g_eats**2)))
    
    ############################
    #### updating functions ####
    ############################
    def update_gamma_inj(self, g_eats):
        # eq. 13
        return(m_p / m_e * self.ee * (g_eats - 1) + 1)

    def update_gamma_cool(self, g_eats, r_eats, B):
        # eq. 14
        return(15 * np.pi * m_e * cc**2 * np.sqrt(g_eats**2 - 1) / (sT * B**2 * r_eats))
    
    def update_B(self, g_eats):
        # eq. 15
        return(np.sqrt(32 * np.pi * self.eB * m_p * cc**2 * self.n_ism * (g_eats**2 - 1)))

    def update_nu_inj(self, gamma_inj, B):
        # eq. 16
        return((0.25 * e * gamma_inj**2 * B) / (m_e * cc))
    
    def update_nu_cool(self, gamma_cool, B):
        # eq. 16
        return(0.25 * e * gamma_cool**2 * B / (m_e * cc))
    
    def update_Pprime(self, B, gamma_inj):
        # in the paragraph after eq. 18
         return((4 / 3) * sT * cc * (B**2 / (8 * np.pi)) * (gamma_inj**2 - 1))
    
    def update_delta(self, g_eats, b_eats, eff_theta):
        # in the paragraph after eq. 19
        return(1 / (g_eats * (1 - b_eats * np.cos(eff_theta))))
    
    def update_nu_prime(self, delta):
        # not sure where this is from, but it makes sense. davide would know.
        return(self.nu_obs / delta)
    
    def update_Iprime_nu(self, r_eats, nu_cool, nu_inj, nu_prime, Pprime):
        # the logic here is not in Rossi, but it should be.  I just got it from davide.   
        Iprime_nu = np.zeros_like(r_eats)
        
        nu_cool_greater = np.where(nu_cool > nu_inj)
        if nu_cool_greater[0].size>0:
            nu_max = nu_inj
            
            # eq. 19
            Iprime_peak = Pprime * self.n_ism * r_eats / nu_max
            
            # the following equations are not in Rossi.
            nu_p_less = np.where(nu_prime < nu_inj)
            Iprime_nu[nu_p_less] = Iprime_peak[nu_p_less] * (nu_prime[nu_p_less] / nu_inj[nu_p_less])**(1 / 3)
            
            nu_p_mid=np.where((nu_prime >= nu_inj) & (nu_prime < nu_cool))
            Iprime_nu[nu_p_mid] = Iprime_peak[nu_p_mid] * (nu_prime[nu_p_mid] / nu_inj[nu_p_mid])**(-(self.pel-1) / 2)
            
            nu_p_greater=np.where(nu_prime >= nu_cool)
            Iprime_nu[nu_p_greater] = Iprime_peak[nu_p_greater] * (nu_inj[nu_p_greater] / nu_cool[nu_p_greater])**((self.pel-1) / 2) * (nu_prime[nu_p_greater] / nu_cool[nu_p_greater])**(-self.pel / 2)
        

        nu_cool_less = np.where(nu_cool <= nu_inj)
        if nu_cool_less[0].size > 0:
            nu_max = nu_cool
            Iprime_peak = Pprime * self.n_ism * r_eats / nu_max
            
            nu_p_less = np.where(nu_prime <= nu_cool)
            Iprime_nu[nu_p_less] = Iprime_peak[nu_p_less] * (nu_prime[nu_p_less] / nu_cool[nu_p_less])**(1 / 3)
                
            nu_p_mid = np.where((nu_prime <= nu_inj) & (nu_prime > nu_cool))
            Iprime_nu[nu_p_mid] = Iprime_peak[nu_p_mid] * (nu_prime[nu_p_mid] / nu_cool[nu_p_mid])**(-1 / 2)
                
            nu_p_greater=np.where(nu_prime > nu_inj)
            Iprime_nu[nu_p_greater] = Iprime_peak[nu_p_greater] * (nu_cool[nu_p_greater] / nu_inj[nu_p_greater])**(1 / 2) * (nu_prime[nu_p_greater] / nu_inj[nu_p_greater])**(-self.pel / 2)
            
        return(Iprime_nu)
    
    def update_dL_nu(self, r_eats, Iprime_nu, thetas2D, nu_cool, nu_inj, delta):
        # eq. 20
        dL_nu = Iprime_nu * delta**3 * r_eats**2 * np.sin(thetas2D) * self.d_theta * self.d_phi
        
        return(dL_nu)
            
    
    def make_EATS_arr(self):
        '''
        This array holds all the values which are defined over the observed surface.
        Most of these values change over the course of the integration and are recomputed
        at each timestep.
        '''
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
                       ('beta_EATS', 'float'),
                       ('delta', 'float'),
                       ('nu_prime', 'float'),
                       ('Iprime_nu', 'float'),
                       ('dL_nu', 'float')]
                       
        EATS_array = np.zeros([self.n_theta, self.n_phi], dtype = EATS_dtypes)
        return(EATS_array)
    
    def update_EATS_arr(self, EATS_arr):   
        '''
        This just handles executing all of the functions above and putting
        them in their correct arrays.
        ''' 
        EATS_arr['gamma_inj'] = self.update_gamma_inj(EATS_arr['G_EATS'])
        
        EATS_arr['B'] = self.update_B(EATS_arr['G_EATS'])
        
        EATS_arr['nu_inj'] = self.update_nu_inj(EATS_arr['gamma_inj'],
                                                EATS_arr['B'])
        
        EATS_arr['gamma_cool'] = self.update_gamma_cool(EATS_arr['G_EATS'],
                                                        EATS_arr['R_EATS'],
                                                        EATS_arr['B'])
        
        EATS_arr['nu_cool'] = self.update_nu_cool(EATS_arr['gamma_cool'],
                                                  EATS_arr['B'])
        
        EATS_arr['Pprime'] = self.update_Pprime(EATS_arr['B'], EATS_arr['gamma_inj'])
        
        EATS_arr['beta_EATS'] = self.beta_from_gamma(EATS_arr['G_EATS'])
        
        EATS_arr['delta'] = self.update_delta(EATS_arr['G_EATS'],
                                              EATS_arr['beta_EATS'],
                                              EATS_arr['eff_theta'])
        
        EATS_arr['nu_prime'] = self.update_nu_prime(EATS_arr['delta'])
        
        EATS_arr['Iprime_nu'] = self.update_Iprime_nu(EATS_arr['R_EATS'],
                                                      EATS_arr['nu_cool'],
                                                      EATS_arr['nu_inj'],
                                                      EATS_arr['nu_prime'],
                                                      EATS_arr['Pprime'])
        
        EATS_arr['dL_nu'] = self.update_dL_nu(EATS_arr['R_EATS'],
                                              EATS_arr['Iprime_nu'],
                                              EATS_arr['thetas2D'],
                                              EATS_arr['nu_cool'],
                                              EATS_arr['nu_inj'],
                                              EATS_arr['delta'])
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
        time_ind_arr = self.make_time_independent_arr()
        
        ### t_obs is the clock for the lightcurve data
        t_obs = self.get_t_obs(start_time, end_time)
        
        ### this will store the output lightcurve data
        lightcurve = self.new_lightcurve(t_obs)
        
        vec_obs = np.array([np.sin(self.theta_obs),
                            0,
                            np.cos(self.theta_obs)])
        
        # update all the quantities like gamma_inj, B, nu_inj etc...
        EATS_arr = self.make_EATS_arr()
        EATS_arr['thetas2D'] = np.meshgrid(self.th_vals, self.ph_vals)[0].T
        
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
                                                               EATS_arr['eff_theta'][i_theta, i_phi])
                    
                    EATS_arr['R_EATS'][i_theta, i_phi] = np.interp(t_obs[timestep],
                                                                   time_of_flight,
                                                                   self.r_vals)
                    
                    EATS_arr['G_EATS'][i_theta, i_phi] = np.interp(EATS_arr['R_EATS'][i_theta, i_phi],
                                                                   self.r_vals,
                                                                   time_ind_arr['G'][:, i_theta])

            #print(EATS_arr['G_EATS'])
            ### Calculate all the quantities associated with the new surface
            EATS_arr = self.update_EATS_arr(EATS_arr)
            #print(EATS_arr['R_EATS'])
            ### Add the luminosity to the lightcurve at this timestep
            lightcurve[timestep, 1] = EATS_arr['dL_nu'].sum() 
            
            ### Update the screen with the progress
            prog_str = '\r Time evolving... {:.1F}%'
            sys.stdout.write(prog_str.format(timestep / len(t_obs) * 100))
            sys.stdout.flush()
            
        return(lightcurve)
        
    ################################################
    ### convert output lightcurve to other units ###
    ################################################
    def cm_from_z(self, z):
        H_0 = 70 # an approximate value for Hubble's constant in km/s/Mpc
        ckm = cc * 1e-4 # convert c to km/s
        D_Mpc = (z * ckm) / H_0 # distance in Mpc
        D_cm = 3.1e24 * D_Mpc # distance in cm
        return(D_cm)

    def get_Jy_flux(self, lc):
        d = self.cm_from_z(self.z)
        return(lc / (4 * np.pi * d**2 * 1e-23))

    def ab_from_jy(self, Jy):
        muJy = Jy * 1e-6
        ab = (239 / 10) - (5 * np.log10(muJy) / (2 * (np.log10(2) + np.log10(5))))
        return(ab)

    ######################
    #### make a plot! ####
    ######################
    def make_curve(self, data, units = None, ax = None, **kwargs):
        data = np.copy(data)
        # convert time to days
        data[:, 0] /= 86400
        
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        
        jflux = self.get_Jy_flux(data[:, 1])
        abflux = self.ab_from_jy(jflux)
        
        if units == 'Jy':
            data[:, 1] = jflux
        elif units == 'AB':
            data[:, 1] = abflux
         
        lightcurve_plot = ax.loglog(data[:, 0], data[:, 1], **kwargs)
        return(lightcurve_plot)
'''
if __name__ == "__main__":
    # here're some small inputs for a quick test plot
    testcurve = Lightcurve(n_theta = 50, n_phi = 10, dr = 0.1, dt = 0.1)
    data = testcurve.time_evolve()
    testcurve.make_curve(data)
    plt.show()
'''