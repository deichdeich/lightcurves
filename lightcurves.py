"""
Author: Alex Deich
Date: January, 2017

lightcurves.py: Calculates GRB lightcurves after the prescription in Rossi et al. 2004.

Requires altonbrown.py to generate equal arrival time surface.

All units cgs.
"""

from __future__ import division, print_function
import numpy as np
import altonbrown
import matplotlib.pyplot as plt

### Defining global consts. 
cc = 29979245800 # speed of light in cm/s

### Some helpful functions
def gamma_from_beta(beta):
    return(np.sqrt(1/(1 - beta**2)))

def beta_from_gamma(gamma):
    return(np.sqrt(1 - (1/gamma**2)))

def solid_angle(theta):
    return(4 * np.pi * (1 - np.cos(theta)))

# maybe output a dictionary?  want unique mass for each theta
def mass_from_energy(energy, lorentz):
    return(energy/(lorentz * cc**2))
    


### Calculate and time-evolve luminosity
class Lightcurve(object):
    def __init__(self, numbins, dt = 0.001, t_lab = 1):
        
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

        self.e_c = self.E_iso/numbins
        self.dr = cc * self.dt # this isn't really correct.
        self.dTh = (np.pi/2)/numbins
        self.dph = (2 * np.pi)/numbins
        self.numbins = numbins # the resolution of the surface.  At the moment, r, th, and
                               # ph are all binned with this number.
        
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

    
    def initialize(self):
        # eq. 1
        energy_per_omega = self.e_c/(1 + (self.eats[:, 1]/self.T_c)**(self.al_e * self.be_e))**(1/self.be_e)
        # eq. 2
        lorentz_per_omega = self.G_c/(1 + (self.eats[:, 1]/self.T_c)**(self.al_g * self.be_g))**(1/self.be_g)
        
        self.M_0_inv = lorentz_per_omega * cc ** 2 / energy_per_omega
        return(lorentz_per_omega)        
    
    def density(self):
        return(self.m_p * self.n)
    
    def update_gamma(self):
        gamma_0 = self.eats[:,4]
        f = self.eats[:,3]
        numerator = np.sqrt(1 + (4 * gamma_0 * f) + (4 * f**2)) - 1
        denominator = 2 * f
        print("update_gammas numerator:", numerator)
        print("Average:", np.mean(numerator))
        print("update_gammas denominator:", denominator)
        print("Average:", np.mean(denominator))
        gamma = numerator/denominator
        

        if gamma.any() < 1:
            raise ArithmeticError("Gamma less than 1!")
            
        return(gamma)
    
    def update_delta(self):
        gamma = self.eats[:,5]
        print("gamma according to delta:", gamma)
        beta = beta_from_gamma(gamma)
        print("beta according to delta:", beta)
        denominator = gamma * (1 - beta * np.cos(self.eats[:,1]))
        print("denominator for delta:", denominator)
        return(1/denominator)
        
    
    #eq. 5. ratio of swept-up mass to initial fireball rest mass, function of theta.        
    def update_f(self):
        omega = solid_angle(self.eats[:,1])
        rho = self.density()
        df = (self.M_0_inv * omega * rho * self.eats[:,0]**2 * cc * self.dt) # using c*dt here instead of dr for now.
        return(df)
    
    def update_B(self):
        const = np.sqrt(32 * np.pi * self.epsilon_B * self.m_p * cc**2 * self.n)
        otherthing = np.sqrt(self.eats[:,5]**2 - 1)
        
        return(const * otherthing)
    
    def update_g_m(self):
        return(self.m_p / self.m_e * self.epsilon_e * (self.eats[:,5] - 1) )
    
    def update_P(self):
        const = (4 / 3) * self.sigma_T * cc * (1/(8 * np.pi))
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
        
        
    def do_all_the_calcs(self, dark_eats):
        # I'll make an array full of 0's that I'll fill with all the new calculations
        bright_eats = np.zeros_like(self.eats)
        """
        The columns are
        
        0: r
        1: theta
        2: phi
        3: f
        4: G_0
        5: G_j
        6: delta
        7: B
        8: g_m
        9: nu_m
        10: P
        11: I_p
        12: dL
        """
        # The order that these are performed does not matter
        bright_eats[:, 0:3] = dark_eats
        print("After updating coords:", np.isnan(bright_eats).any())
        bright_eats[:, 3] = self.update_f()
        print("After updating f:", np.isnan(bright_eats).any())
        print("Max of f:", np.max(bright_eats[:,3]))
        bright_eats[:, 4] = self.eats[:,4]
        print("After adding G_0:", np.isnan(bright_eats).any())
        bright_eats[:, 5] = self.update_gamma()
        print("After updating G:", np.isnan(bright_eats).any())
        bright_eats[:, 6] = self.update_delta()
        print("After updating delta:", np.isnan(bright_eats).any())
        bright_eats[:, 7] = self.update_B()
        print("After updating B:", np.isnan(bright_eats).any())
        bright_eats[:, 8] = self.update_g_m()
        print("After updating g_m:", np.isnan(bright_eats).any())
        bright_eats[:, 9] = self.update_nu_m()
        print("After updating nu_m:", np.isnan(bright_eats).any())
        bright_eats[:, 10] = self.update_P()
        print("After updating P:", np.isnan(bright_eats).any())
        bright_eats[:, 11] = self.update_I_p()
        print("After updating I_p:", np.isnan(bright_eats).any())
        bright_eats[:, 12] = self.update_dL()
        print("After updating dL:", np.isnan(bright_eats).any())
        
        return(bright_eats)
        
        
        
    
    def time_evolve(self, nsteps):
        # for every timestep, pass the output of altonbrown() to
        # do_all_the_calcs(), whose output is all of the data for the current
        # timestep.  Then, sum the intensity column and record that sum in the
        # lightcurve array.
        
        self.lightcurve = np.zeros((nsteps,2))

        r_lim1 = altonbrown.get_rlim(G_sh = 1e4,
                                     t = self.t_lab,
                                     r_dec = 1e16,
                                     alpha = 0,
                                     delta = 0)

        init_dark_eats = altonbrown.good_eats(G_sh = 1e4,
                                              t = self.t_lab,
                                              r_dec = 1e16,
                                              r_lim = r_lim1,
                                              numbins = self.numbins,
                                              alpha = 0,
                                              delta = 0)
                                              
                                              
        self.eats = np.zeros((len(init_dark_eats), 13))
        
        self.eats[:,0:3] = init_dark_eats
        self.eats[:, 4] = self.initialize()
        self.eats[:, 5] = self.initialize()
        
        for step in xrange(1,nsteps):
            print(step,"\n\n")
            print("G_0:", self.eats[0:10,4])
            new_r_lim = altonbrown.get_rlim(G_sh = self.eats[0,4],
                                            t = self.t_lab,
                                            r_dec = 1e16,
                                            alpha = 0,
                                            delta = 0)
            print("rlim:",new_r_lim)                            
            new_eats = altonbrown.good_eats(G_sh = self.eats[0,4],
                                            t = self.t_lab,
                                            r_dec = 1e16,
                                            r_lim = new_r_lim,
                                            numbins = self.numbins,
                                            alpha = 0,
                                            delta = 0)
                                            
            self.eats = self.do_all_the_calcs(new_eats)

            
            self.lightcurve[step,0] = np.nansum(self.eats[:,12])
            self.lightcurve[step,1] = self.t_lab
            self.t_lab += self.dt
        
    
    def plot_lightcurve(self, savefig = False, fname = "lightcurve.pdf"):
        plt.plot(self.lightcurve[:,0],self.lightcurve[:,1])
        if savefig == False:
            plt.show()
        elif savefig == True:
            plt.savefig(fname)
    
    def plot_3d_heatmap(self, savefig = False, fname = "heatmap.pdf"):
        
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        
        ax = fig.gca(projection = '3d')
        
        p = ax.scatter(self.eats[:,0] * np.sin(self.eats[:,1]) * np.cos(self.eats[:,2]),
                self.eats[:,0] * np.cos(self.eats[:,1]),
                self.eats[:,0] * np.sin(self.eats[:,1]) * np.sin(self.eats[:,2]),
                c = np.nan_to_num(self.eats[:,12]),
                alpha = 0.3)
        
        plt.colorbar(p)
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        #ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.zaxis.set_major_formatter(plt.NullFormatter())
        plt.show()
        
        
        
if __name__ == "__main__":
    test_curve = Lightcurve(40)
    test_curve.time_evolve(2)
    print(test_curve.lightcurve)
    #test_curve.plot_3d_heatmap()
    #test_curve.plot_lightcurve()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        