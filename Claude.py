"""CLimate Analysis using Digital Estimations (CLAuDE)

TODO(): add proper documentation here
"""

### !!!!!!!!! ###
import warnings
warnings.filterwarnings("ignore")
### !!!!!!!!! ###

# preinstalled libs
import sys
import time
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

from copy import deepcopy

# Local libs
import claude_low_level_library as low_level
import claude_top_level_library as top_level

from util import *
from Settings import Settings
from ClaudeBackend import ClaudeBackend

class Claude(ClaudeBackend):
    def run(self):
        try:
            self.__run()
        except KeyboardInterrupt as kbi:
            print("Quitting gracefully...")
            sys.exit(0)

    def __run(self) -> None:
        while True: # probably don't do thiself...
            self.initial_time = time.time()

            if self.t < self.spinup_length:
                self.dt = self.dt_spinup
                self.velocity = False
            else:
                self.dt = self.dt_main
                self.velocity = True

            # print current time in simulation to command line
            if self.verbose:
                print("+++ t = " + str(round(self.t/self.day,2)) + " days +++")
                print('T: ',
                      round(self.temperature_world.max()-273.15,1),
                      ' - ',
                      round(self.temperature_world.min()-273.15,1),
                      ' C')
                print('U: ',
                      round(self.u[:,:,:self.sponge_index-1].max(),2),
                      ' - ',
                      round(self.u[:,:,:self.sponge_index-1].min(),2),
                      ' V: ',
                      round(self.v[:,:,:self.sponge_index-1].max(),2),
                      ' - ',
                      round(self.v[:,:,:self.sponge_index-1].min(),2),
                      ' W: ',
                      round(self.w[:,:,:self.sponge_index-1].max(),2),
                      ' - ',
                      round(self.w[:,:,:self.sponge_index-1].min(),4))

            self.tracer[40,50,self.sample_level] = 1
            self.tracer[20,50,self.sample_level] = 1

            if self.verbose: self.before_radiation = time.time()
            tmp_temps = top_level.radiation_calculation(self.temperature_world, 
                                                        self.potential_temperature, 
                                                        self.pressure_levels, 
                                                        self.heat_capacity_earth, 
                                                        self.albedo, 
                                                        self.insolation, 
                                                        self.lat, 
                                                        self.lon, 
                                                        self.t, 
                                                        self.dt, 
                                                        self.day, 
                                                        self.year, 
                                                        self.axial_tilt)
 
            self.temperature_world, self.potential_temperature = tmp_temps[0], tmp_temps[1]

            if self.smoothing: 
                self.potential_temperature = top_level.smoothing_3D(self.potential_temperature,self.smoothing_parameter_t)
            
            if self.verbose:
                self.time_taken = float(round(time.time() - self.before_radiation,3))
                print('Radiation: ',str(self.time_taken),'s')

            self.diffusion = top_level.laplacian_2d(self.temperature_world,self.dx,self.dy)
            self.diffusion[0,:] = np.mean(self.diffusion[1,:],axis=0)
            self.diffusion[-1,:] = np.mean(self.diffusion[-2,:],axis=0)
            self.temperature_world -= self.dt*1E-5*self.diffusion

            # update geopotential field
            self.geopotential = np.zeros_like(self.potential_temperature)
            for k in np.arange(1,self.nlevels):  
                self.geopotential[:,:,k] = self.geopotential[:,:,k-1] - self.potential_temperature[:,:,k]*(self.sigma[k]-self.sigma[k-1])

            if self.velocity:
                if self.verbose: 
                    self.before_velocity = time.time()
                
                self.u_add,self.v_add = top_level.velocity_calculation(self.u,
                                                                       self.v,
                                                                       self.w,
                                                                       self.pressure_levels,
                                                                       self.geopotential,
                                                                       self.potential_temperature,
                                                                       self.coriolis,
                                                                       self.gravity,
                                                                       self.dx,
                                                                       self.dy,
                                                                       self.dt,
                                                                       self.sponge_index)

                if self.verbose: 
                    self.time_taken = float(round(time.time() - self.before_velocity,3))
                    print('Velocity: ',str(self.time_taken),'s')

                if self.verbose: 
                    self.before_projection = time.time()
                
                self.grid_velocities = (self.x_dot_N,self.y_dot_N,self.x_dot_S,self.y_dot_S)
            
                tmp_planes = top_level.polar_planes(self.u,
                                                    self.v,
                                                    self.u_add,
                                                    self.v_add,
                                                    self.potential_temperature,
                                                    self.geopotential,
                                                    self.grid_velocities,
                                                    self.indices,
                                                    self.grids,
                                                    self.coords,
                                                    self.coriolis_plane_N,
                                                    self.coriolis_plane_S,
                                                    self.grid_side_length,
                                                    self.pressure_levels,
                                                    self.lat,
                                                    self.lon,
                                                    self.dt,
                                                    self.polar_grid_resolution,
                                                    self.gravity,
                                                    self.sponge_index)

                self.u_add  , self.v_add   = tmp_planes[0], tmp_planes[1]
                self.x_dot_N, self.y_dot_N = tmp_planes[2], tmp_planes[3]
                self.x_dot_S, self.y_dot_S = tmp_planes[4], tmp_planes[5]
                
                self.u += self.u_add
                self.v += self.v_add

                if self.smoothing: self.u = top_level.smoothing_3D(self.u,self.smoothing_parameter_u)
                if self.smoothing: self.v = top_level.smoothing_3D(self.v,self.smoothing_parameter_v)

                tmp_vels = top_level.update_plane_velocities(self.lat,
                                                             self.lon,
                                                             self.pole_low_index_N,
                                                             self.pole_low_index_S,
                                                             np.flip(self.u[self.pole_low_index_N:,:,:],axis=1),
                                                             np.flip(self.v[self.pole_low_index_N:,:,:],axis=1),
                                                             self.grids,
                                                             self.grid_lat_coords_N,
                                                             self.grid_lon_coords_N,
                                                             self.u[:self.pole_low_index_S,:,:],
                                                             self.v[:self.pole_low_index_S,:,:],
                                                             self.grid_lat_coords_S,
                                                             self.grid_lon_coords_S)

                self.x_dot_N, self.y_dot_N = tmp_vels[0], tmp_vels[1]
                self.x_dot_S, self.y_dot_S = tmp_vels[2], tmp_vels[3]

                self.grid_velocities = (self.x_dot_N,self.y_dot_N,self.x_dot_S,self.y_dot_S)
                
                if self.verbose: 
                    self.time_taken = float(round(time.time() - self.before_projection,3))
                    print('Projection: ',str(self.time_taken),'s')

                ### allow for thermal advection in the atmosphere
                if self.verbose: self.before_advection = time.time()

                if self.verbose: self.before_w = time.time()
                # using updated u,v fields calculated w
                # https://www.sjsu.edu/faculty/watkins/omega.htm
                self.w = -top_level.w_calculation(self.u,
                                                  self.v,
                                                  self.w,
                                                  self.pressure_levels,
                                                  self.geopotential,
                                                  self.potential_temperature,
                                                  self.coriolis,
                                                  self.gravity,
                                                  self.dx,
                                                  self.dy,
                                                  self.dt,
                                                  self.indices,
                                                  self.coords,
                                                  self.grids,
                                                  self.grid_velocities,
                                                  self.polar_grid_resolution,
                                                  self.lat,
                                                  self.lon)
                if self.smoothing: 
                    self.w = top_level.smoothing_3D(self.w,self.smoothing_parameter_w,0.25)

                self.w[:,:,18:] *= 0
                # w[:1,:,:] *= 0
                # w[-1:,:,:] *= 0

                # plt.semilogy(w[5,25,:sponge_index],pressure_levels[:sponge_index])
                # plt.gca().invert_yaxis()
                # plt.show()

                if self.verbose: 
                    self.time_taken = float(round(time.time() - self.before_w,3))
                    print('Calculate w: ',str(self.time_taken),'s')

                #################################
                
                self.atmosp_addition = top_level.divergence_with_scalar(self.potential_temperature,
                                                                        self.u,
                                                                        self.v,
                                                                        self.w,
                                                                        self.dx,
                                                                        self.dy,
                                                                        self.lat,
                                                                        self.lon,
                                                                        self.pressure_levels,
                                                                        self.polar_grid_resolution,
                                                                        self.indices,
                                                                        self.coords,
                                                                        self.grids,
                                                                        self.grid_velocities)

                if self.smoothing: self.atmosp_addition = top_level.smoothing_3D(self.atmosp_addition,self.smoothing_parameter_add)

                self.atmosp_addition[:,:,self.sponge_index-1] *= 0.5
                self.atmosp_addition[:,:,self.sponge_index:] *= 0

                self.potential_temperature -= self.dt*self.atmosp_addition

                ###################################################################

                self.tracer_addition = top_level.divergence_with_scalar(self.tracer,
                                                                        self.u,
                                                                        self.v,
                                                                        self.w,
                                                                        self.dx,
                                                                        self.dy,
                                                                        self.lat,
                                                                        self.lon,
                                                                        self.pressure_levels,
                                                                        self.polar_grid_resolution,
                                                                        self.indices,
                                                                        self.coords,
                                                                        self.grids,
                                                                        self.grid_velocities)
                self.tracer -= self.dt*self.tracer_addition

                self.diffusion = top_level.laplacian_3d(self.potential_temperature,self.dx,self.dy,self.pressure_levels)
                self.diffusion[0,:,:] = np.mean(self.diffusion[1,:,:],axis=0)
                self.diffusion[-1,:,:] = np.mean(self.diffusion[-2,:,:],axis=0)
                self.potential_temperature -= self.dt*1E-4*self.diffusion

                self.courant = self.w*self.dt
                for k in range(self.nlevels-1):
                    self.courant[:,:,k] /= (self.pressure_levels[k+1] - self.pressure_levels[k])
                if self.verbose:
                    print('Courant max: ',round(abs(self.courant).max(),3))

                ###################################################################

                if self.verbose: 
                    self.time_taken = float(round(time.time() - self.before_advection,3))
                    print('Advection: ',str(self.time_taken),'s')

            if self.t-self.last_plot >= self.plot_frequency*self.dt:
                self._plotting_routine()
                self.last_plot = self.t

            if self.save:
                if self.t-self.last_save >= self.save_frequency*self.dt:
                    pickle.dump((self.potential_temperature,
                                 self.temperature_world,
                                 self.u,
                                 self.v,
                                 self.w,
                                 self.x_dot_N,
                                 self.y_dot_N,
                                 self.x_dot_S,
                                 self.y_dot_S,
                                 self.t,
                                 self.albedo,
                                 self.tracer), open("save_file.p","wb"))
                    self.last_save = self.t

            if np.isnan(self.u.max()):
                sys.exit()

            # advance time by one timestep
            self.t += self.dt

            self.time_taken = float(round(time.time() - self.initial_time,3))

            if self.verbose:
                print('Time: ',str(self.time_taken),'s')
