import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
from scipy.misc import derivative
from scipy.interpolate import CubicSpline
from scipy import signal
from matplotlib.animation import PillowWriter

plt.style.use('default')

base_file_path = r'C:\\Users\\marti\\OneDrive\\Documents\\ICLR\\Hot Fire Test Data\\Test_Data'

# Class to contain all the test data from different tests and methods to process it
class Data:
    def __init__(self, file_name, 
                 file_path, 
                 test_name, 
                 t_start, 
                 t_ignition, 
                 t_end, 
                 source, 
                 ch8name):
        
        self.name = test_name
        self.file = file_name
        self.path = os.path.join(base_file_path,file_path)
        self.t_start = t_start
        self.t_end = t_end
        self.t_ign = t_ignition
        self.ch8name = ch8name
        
        self.data = pd.read_csv(os.path.join(base_file_path,file_path,file_name), skip_blank_lines=True)
        
        self.data.dropna(inplace=True)
        
        
        self.data.rename(columns={'ch0sens': 'P_N2_Tank',
                             'ch2sens': 'P_Chamber',
                             'ch3sens': 'P_Fuel_Tank',
                             'ch6sens': 'Ox_mass',
                             'ch7sens': 'Thrust',
                             'ch8sens': ch8name, #P_Fuel_Inlet, P_Ox_Inj etc.
                             'ch9sens': 'P_Ox_Tank',
                             'ch10sens': 'P_Fuel_Injector',
                             'temp0': 'TC1',
                             'temp1': 'TC2',
                             'temp2': 'Fuel_Flow_Rate',
                             'temp3': 'TC3',
                             'system_time': 'Time'}, inplace=True)
        
        if source == 'BACKEND':
            self.data.loc[:,'Time'] = self.data.loc[:,'Time']/1000
        else:
            self.data.loc[:,'Time'] = self.data.loc[:,'Time']/1000000
        
        Time = self.data.loc[:,'Time']
        #self.data.loc[:,'Time'] = self.data.loc[:,'Time'] - Time.iloc[0]
        
        self.data = self.data.loc[self.data['Time'] > t_start]
        self.data = self.data.loc[self.data['Time'] < t_end]
        self.data.loc[:,'Time'] = self.data.loc[:,'Time'] - t_ignition
        
        self.data.loc[:,'Thrust'] = -self.data.loc[:,'Thrust']
        
    def smoothed_thrust(self):
        self.Smoothed_thrust = gaussian_filter(self.data.loc[:,'Thrust'],sigma=1)
        return self.Smoothed_thrust
    
    def ox_mass(self, mass_offset):
        b, a = signal.butter(4, 0.3)
        ox_mass = (-self.data.loc[:,'Ox_mass']/9.81) + mass_offset
        ox_mass = signal.filtfilt(b, a, ox_mass)
        return ox_mass
    
    def impulse(self):
        Impulse = integrate.simpson(self.data.loc[:,'Thrust'], self.data.loc[:,'Time'])
        print('Total Impulse: ', Impulse, ' Ns')
    
    def CF_impulse(self, CF_A):
        Impulse = integrate.simpson(self.data.loc[:,'P_Chamber']*CF_A*1e5, self.data.loc[:,'Time'])
        print('Total Impulse: ', Impulse, ' Ns')
    
    def fuel_mass_flow(self, rho_fuel):
        b, a = signal.butter(4, 0.1)
        self.m_dot_fuel = rho_fuel*signal.filtfilt(b, a, self.data.loc[:,'Fuel_Flow_Rate'])/1000
        return self.m_dot_fuel
    
    def fuel_mass_flow_inj_dp(self):
        CdA_inj = 4.9e-5
        #CdA_inj = 7e-5
        rho_fuel = 850
        dp_Pa = (self.data.loc[:,'P_Fuel_Injector'] - self.data.loc[:,'P_Chamber'])*1e5
        m_dot_fuel = CdA_inj*(rho_fuel*dp_Pa)/np.sqrt(abs(rho_fuel*dp_Pa))
        return m_dot_fuel
    
    def ox_mass_flow(self):
        cs = CubicSpline(self.data.loc[:,'Time'],self.ox_mass(-11),bc_type='clamped')
        self.m_dot_ox = -derivative(cs, self.data.loc[:,'Time'])
        return self.m_dot_ox
    
    def ox_mass_from_c_star(self):
        ox_mass = integrate.simpson(self.ox_mass_flow_c_star(), self.data.loc[:,'Time'])
        print('Total Ox Mass: ', ox_mass, ' kg')
    
    def c_star(self, OF, combustion_eff):
        #c_star = (-22.912*OF**2 + 222.6*OF + 913.89)*combustion_eff
        c_star = (-31.514*OF**2 + 272.3*OF + 903.49)*combustion_eff
        return c_star
    
    def ox_mass_flow_c_star(self):
        N = 10 #number of calc iterations
        combustion_eff = 1
        rho_fuel = 850
        D_throat = 42.5e-3
        A_throat = (np.pi*D_throat**2)/4
        OF = 1.5
        for i in range(0, N):
            m_dot_tot = A_throat*self.data.loc[:,'P_Chamber']*1e5/self.c_star(OF, combustion_eff)
            m_dot_ox = m_dot_tot - self.fuel_mass_flow_inj_dp()
            OF = m_dot_ox/self.fuel_mass_flow_inj_dp()
        
        return m_dot_ox
    
    def isp(self):
        isp = abs(self.data.loc[:,'Thrust'])/(9.81*(self.ox_mass_flow_c_star()+self.fuel_mass_flow_inj_dp()))
        return isp

    def low_pressure_plot(self):
        plt.plot(self.data.loc[:,'Time'],self.data.loc[:,'P_Ox_Tank'], label = 'Ox Tank')
        plt.plot(self.data.loc[:,'Time'],self.data.loc[:,'P_Fuel_Tank'], label = 'Fuel Tank')
        plt.plot(self.data.loc[:,'Time'],self.data.loc[:,self.ch8name], label = self.ch8name)
        plt.plot(self.data.loc[:,'Time'],self.data.loc[:,'P_Fuel_Injector'], label = 'Fuel Injector')
        plt.plot(self.data.loc[:,'Time'],self.data.loc[:,'P_Chamber'], label = 'Chamber')

        plt.grid(which='major',axis='both',linewidth = 0.8)
        plt.minorticks_on()
        plt.grid(which='minor',axis='both',linewidth = 0.2)
        plt.xlabel('Time [s]')
        plt.ylabel('Pressure [bar]')
        plt.legend()
        plt.title(self.name +'_Low-Pressures')
        filename = self.name + '_Low-Pressures.png'
        plt.savefig(os.path.join(self.path,filename), dpi=300)
        plt.show()
    
    def chamber_pressure_plot(self):
        #plt.plot(self.data.loc[:,'Time'],self.data.loc[:,self.ch8name], label = self.ch8name)
        #plt.plot(self.data.loc[:,'Time'],self.data.loc[:,'P_Fuel_Injector'], label = 'Fuel Injector')
        plt.plot(self.data.loc[:,'Time'],self.data.loc[:,'P_Chamber'], label = 'Chamber')

        plt.grid(which='major',axis='both',linewidth = 0.8)
        plt.minorticks_on()
        plt.grid(which='minor',axis='both',linewidth = 0.2)
        plt.xlabel('Time [s]')
        plt.ylabel('Pressure [bar]')
        plt.legend()
        plt.title(self.name +'_Engine-Pressures')
        filename = self.name + '_Engine-Pressures.png'
        plt.savefig(os.path.join(self.path,filename), dpi=300)
        plt.show()
    
    def high_pressure_plot(self):
        
        plt.plot(self.data.loc[:,'Time'],self.data.loc[:,'P_N2_Tank'], label = 'HP Tank')
        #plt.plot(self.data.loc[:,'Time'],self.data.loc[:,'P_Ox_Tank'], label = 'Ox Tank')
        plt.plot(self.data.loc[:,'Time'],self.data.loc[:,'P_Fuel_Tank'], label = 'Fuel Tank')
        #plt.plot(self.data.loc[:,'Time'],self.data.loc[:,self.ch8name], label = self.ch8name)
        plt.plot(self.data.loc[:,'Time'],self.data.loc[:,'P_Fuel_Injector'], label = 'Fuel Injector')
        #plt.plot(self.data.loc[:,'Time'],self.data.loc[:,'P_Chamber'], label = 'Chamber')

        plt.grid(which='major',axis='both',linewidth = 0.8)
        plt.minorticks_on()
        plt.grid(which='minor',axis='both',linewidth = 0.2)
        plt.xlabel('Time [s]')
        plt.ylabel('Pressure [bar]')
        plt.legend()
        plt.title(self.name +'_High-Pressures')
        filename = self.name + '_High-Pressures.png'
        plt.savefig(os.path.join(self.path,filename), dpi=300)
        plt.show()
    
    def thrust_curve_plot(self):
        plt.plot(self.data.loc[:,'Time'],abs(self.smoothed_thrust()), label = 'Smoothed_Thrust',color='black')
        plt.scatter(self.data.loc[:,'Time'],abs(self.data.loc[:,'Thrust']), label = 'Raw Thrust', marker = 'o', color='red', s=1)

        plt.grid(which='major',axis='both',linewidth = 0.8)
        plt.minorticks_on()
        plt.grid(which='minor',axis='both',linewidth = 0.2)
        plt.xlabel('Time [s]')
        plt.ylabel('Thrust [N]')
        #plt.xlim(1.5,11.6)
        plt.legend()
        plt.title(self.name +'_Thrust-Curve')
        filename = self.name +'_Thrust-Curve.png'
        plt.savefig(os.path.join(self.path,filename), dpi=300)
        plt.show()
        
    def thrust_curve_plot_CPLC(self, t_lower, t_upper):
        
        #----------------------------------------------------------------
        #--------------------- Parameter Definition ---------------------
        #----------------------------------------------------------------

        T_nominal = 2400 #Nominal high thrust [N]
        T_startup = 500 #Startup thrust
        Percent_tolerance = 5 
        T_tolerance = T_nominal*(Percent_tolerance/100)
        Percent_low_throttle = 40
        T_low = (T_nominal - T_tolerance) * (Percent_low_throttle/100) - T_tolerance

        g = 9.81

        Isp = 190 #Specific impulse in seconds

        N2O_mass = 7.5 #Maximum nitrous mass [kg]

        T_gradient_1 = 10000 #Gradient of 1st throttle-up rate of change of thrust [N/s]
        T_gradient = 1700 #Gradient of rate of change of thrust [N/s]
        #Combinations: T_grad = 1300, t_low = 2.5; T_grad = 1050, t_low = 2

        t_duration = 12
        dt = 0.01

        t = np.arange(0, t_duration + dt, dt)

        t_throttle_up_1 = 0.1
        t_high_thrust = 4.3
        t_low_thrust = 3


        #----------------------------------------------------------------
        #-------------------- Create Thrust Profile ---------------------
        #----------------------------------------------------------------
        
        def thrust_trace(offset):
            t_throttle_down = t_throttle_up_1 + (T_nominal-T_startup)/T_gradient_1 + t_high_thrust + offset

            t_throttle_up_2 = t_throttle_down + (T_nominal-T_low)/T_gradient + t_low_thrust - offset*2

            T = np.zeros(len(t))
            for i in range(0,len(t)):
                if t[i] < t_throttle_up_1: #Startup
                    T[i] = T_startup
                elif t[i] < (t_throttle_up_1 + (T_nominal-T_startup)/T_gradient_1) : #Throttle Up 1
                    T[i] = T_startup + T_gradient_1*(t[i] - t_throttle_up_1)
                elif t[i] < (t_throttle_down): #High Thrust
                    T[i] = T_nominal
                elif t[i] < (t_throttle_down + (T_nominal-T_low)/T_gradient): #Throttle Down
                    T[i] = T_nominal - (t[i] - t_throttle_down) * T_gradient
                elif t[i] < (t_throttle_down + (T_nominal-T_low)/T_gradient + t_low_thrust - offset*2): #Low Thrust
                    T[i] = T_low
                elif t[i] < (t_throttle_up_2 + (T_nominal-T_low)/T_gradient): #Throttle Up 2
                    T[i] = T_low + T_gradient*(t[i] - t_throttle_up_2)
                else:
                    T[i] = T_nominal
            return T
        
        T = thrust_trace(0)
        T_max = thrust_trace(0.15) + T_tolerance
        T_min = thrust_trace(-0.15) - T_tolerance
        
        plt.xlim(t_lower,t_upper)
        
        plt.plot(t,T, label = 'Nominal_Thrust_Trace',color='blue')
        plt.plot(t,T_max, label = 'Max_Thrust_Trace',color='orange')
        plt.plot(t,T_min, label = 'Min_Thrust_Trace',color='green')
        self.thrust_curve_plot()
        
    def animated_thrust_trace(self):
        
        cs = CubicSpline(self.data.loc[:,'Time'],-self.data.loc[:,'Thrust'],bc_type='clamped')
        ts = np.arange(-2,14,0.04)
        
        #----------------------------------------------------------------
        #--------------------- Parameter Definition ---------------------
        #----------------------------------------------------------------

        T_nominal = 2400 #Nominal high thrust [N]
        T_startup = 0 #Startup thrust
        Percent_tolerance = 5 
        T_tolerance = T_nominal*(Percent_tolerance/100)
        Percent_low_throttle = 40
        T_low = (T_nominal - T_tolerance) * (Percent_low_throttle/100) - T_tolerance

        g = 9.81

        Isp = 190 #Specific impulse in seconds

        N2O_mass = 7.5 #Maximum nitrous mass [kg]

        T_gradient_1 = 10000 #Gradient of 1st throttle-up rate of change of thrust [N/s]
        T_gradient = 1700 #Gradient of rate of change of thrust [N/s]
        #Combinations: T_grad = 1300, t_low = 2.5; T_grad = 1050, t_low = 2


        t_throttle_up_1 = 0.1
        t_high_thrust = 4.3
        t_low_thrust = 3


        #----------------------------------------------------------------
        #-------------------- Create Thrust Profile ---------------------
        #----------------------------------------------------------------
        
        def thrust_trace(offset):
            t_throttle_down = t_throttle_up_1 + (T_nominal-T_startup)/T_gradient_1 + t_high_thrust + offset

            t_throttle_up_2 = t_throttle_down + (T_nominal-T_low)/T_gradient + t_low_thrust - offset*2

            T = np.zeros(len(ts))
            for i in range(0,len(ts)):
                if ts[i] < t_throttle_up_1: #Startup
                    T[i] = T_startup
                elif ts[i] < (t_throttle_up_1 + (T_nominal-T_startup)/T_gradient_1) : #Throttle Up 1
                    T[i] = T_startup + T_gradient_1*(ts[i] - t_throttle_up_1)
                elif ts[i] < (t_throttle_down): #High Thrust
                    T[i] = T_nominal
                elif ts[i] < (t_throttle_down + (T_nominal-T_low)/T_gradient): #Throttle Down
                    T[i] = T_nominal - (ts[i] - t_throttle_down) * T_gradient
                elif ts[i] < (t_throttle_down + (T_nominal-T_low)/T_gradient + t_low_thrust - offset*2): #Low Thrust
                    T[i] = T_low
                elif ts[i] < (t_throttle_up_2 + (T_nominal-T_low)/T_gradient): #Throttle Up 2
                    T[i] = T_low + T_gradient*(ts[i] - t_throttle_up_2)
                else:
                    T[i] = T_nominal
            return T
        
        T = thrust_trace(0)
        
        fig = plt.figure()
        l, = plt.plot([], [], 'red', label='Thrust')
        #l2, = plt.plot([], [], 'cyan', label='Set Point')
        #p, = plt.scatter([],[], 'red')
        #p2, = plt.scatter([], [], 'blue')
        #plt.plot(ts, cs(ts))
        plt.grid(which='major',axis='both',linewidth = 0.8)
        plt.minorticks_on()
        #plt.grid(which='minor',axis='both',linewidth = 0.2)
        plt.xlabel('Time [s]')
        plt.ylabel('Thrust [N]')
        plt.xlim(-2,14)
        plt.ylim(-100,3500)
        #plt.legend()
        #plt.title(self.name +'_Thrust-Curve')
        
        
        xlist = []
        ylist = []
        y2list = []
        
        writer = PillowWriter(fps=25)
        
        filename = self.name +'_Animated-Thrust-Curve-2.gif'
        with writer.saving(fig, os.path.join(base_file_path,self.path,filename), 400):
            for i in range(0,len(ts)):
                xlist.append(ts[i])
                ylist.append(cs(ts[i]))
                y2list.append(T[i])
                l.set_data(xlist,ylist)
                #l2.set_data(xlist,y2list)
                #p.set_data(ts[i],cs(ts[i]))
                #p2.set_data(ts[i],T[i])
                writer.grab_frame()
                print('Frame: ', i)
        
    
    def CF_thrust_curve_plot(self, CF_A):
        plt.plot(self.data.loc[:,'Time'],self.data.loc[:,'P_Chamber']*CF_A*1e5, label = 'Calc. from Pc',color='black')
        plt.plot(self.data.loc[:,'Time'],self.smoothed_thrust()+100, label = 'Smoothed_Thrust',color='red')

        plt.grid(which='major',axis='both',linewidth = 0.8)
        plt.minorticks_on()
        plt.grid(which='minor',axis='both',linewidth = 0.2)
        plt.xlabel('Time [s]')
        plt.ylabel('Thrust [N]')
        plt.legend()
        plt.title(self.name +'_CF-Thrust-Curve')
        filename = self.name +'_CF-Thrust-Curve.png'
        plt.savefig(os.path.join(self.path,filename), dpi=300)
        plt.show()
        
    def CF_thrust_curve_plot_lbf(self, CF_A):
        plt.plot(self.data.loc[:,'Time'],0.224809*self.data.loc[:,'P_Chamber']*CF_A*1e5, label = 'Calc. from Pc',color='black')
        plt.plot(self.data.loc[:,'Time'],0.224809*(self.smoothed_thrust()+100), label = 'Load Cell Data',color='red')

        plt.grid(which='major',axis='both',linewidth = 0.8)
        plt.minorticks_on()
        plt.grid(which='minor',axis='both',linewidth = 0.2)
        plt.xlabel('Time [s]')
        plt.ylabel('Thrust [lbf]')
        plt.legend()
        plt.title(self.name +'_CF-Thrust-Curve')
        filename = self.name +'_CF-Thrust-Curve.png'
        plt.savefig(os.path.join(self.path,filename), dpi=300)
        plt.show()
        
    def fuel_flow_vs_pressure(self):
        self.Smoothed_Flow_Rate = gaussian_filter(self.data.loc[:,'Fuel_Flow_Rate'], sigma=2)

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Pressure [bar]')
        ax1.plot(self.data.loc[:,'Time'],self.data.loc[:,'P_Fuel_Tank'], label = 'Tank Pressure')
        ax1.plot(self.data.loc[:,'Time'],self.data.loc[:,'P_Fuel_Injector'], label = 'Injector Pressure')
        #ax1.plot(self.data.loc[:,'Time'],self.data.loc[:,'P_Chamber'], label = 'Chamber')
        ax1.tick_params(axis='y')
        #ax1.legend()


        plt.grid(which='major',axis='both',linewidth = 0.8)
        plt.minorticks_on()
        plt.grid(which='minor',axis='both',linewidth = 0.2)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'black'
        ax2.set_ylabel('Volumetric Flow Rate [L/s]', color=color)  # we already handled the x-label with ax1
        ax2.plot(self.data.loc[:,'Time'],self.Smoothed_Flow_Rate, label = 'Smoothed Fuel Flow Rate', color='black')
        ax2.scatter(self.data.loc[:,'Time'],self.data.loc[:,'Fuel_Flow_Rate'], label = 'Fuel Flow Rate', marker = 'o', color='red', s=1)
        ax2.set_ylim(0,0.5)
        ax2.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc=4)
        ax2.legend(loc=5)

        fig.tight_layout()# otherwise the right y-label is slightly clipped
        plt.title(self.name + '_Flow-Rate-vs-Pressure')
        filename = self.name +'_Flow-Rate-vs-Pressure.png'
        plt.savefig(os.path.join(self.path,filename), dpi=300) #Add correct file path
        plt.show()
        
    def temperature_plot(self, labels):
        plt.plot(self.data.loc[:,'Time'],self.data.loc[:,'TC1'], label = labels[0])
        #plt.plot(self.data.loc[:,'Time'],self.data.loc[:,'TC2'], label = labels[1])
        #plt.plot(self.data.loc[:,'Time'],self.data.loc[:,'TC3'], label = labels[2])

        plt.grid(which='major',axis='both',linewidth = 0.8)
        plt.minorticks_on()
        plt.grid(which='minor',axis='both',linewidth = 0.2)
        plt.xlabel('Time [s]')
        plt.ylabel('Temperature [deg C]')
        plt.legend()
        plt.title(self.name +'_Temperatures')
        filename = self.name +'_Temperatures.png'
        plt.savefig(os.path.join(self.path,filename), dpi=300)
        plt.show()
        
    def ox_mass_plot(self, mass_offset):
        plt.plot(self.data.loc[:,'Time'],self.ox_mass(mass_offset))
        
        plt.grid(which='major',axis='both',linewidth = 0.8)
        plt.minorticks_on()
        plt.grid(which='minor',axis='both',linewidth = 0.2)
        plt.xlabel('Time [s]')
        plt.ylabel('Oxidiser Mass [kg]')
        plt.legend()
        plt.title(self.name +'_Ox_Mass')
        filename = self.name +'_Ox_Mass.png'
        plt.savefig(os.path.join(self.path,filename), dpi=300)
        plt.show()
        
    def m_dot_prop_plot(self):
        rho_fuel = 850
        
        plt.plot(self.data.loc[:,'Time'],self.ox_mass_flow(), label='Oxidiser load cell derivative')
        plt.plot(self.data.loc[:,'Time'],self.ox_mass_flow_c_star(), label='Ox c_star calculated')
        plt.plot(self.data.loc[:,'Time'],self.fuel_mass_flow(rho_fuel), label='Fuel flow meter')
        plt.plot(self.data.loc[:,'Time'],self.fuel_mass_flow_inj_dp(), label='Fuel injector dp based')
        
        plt.grid(which='major',axis='both',linewidth = 0.8)
        plt.minorticks_on()
        plt.grid(which='minor',axis='both',linewidth = 0.2)
        plt.xlabel('Time [s]')
        plt.ylabel('Mass Flow Rate [kg/s]')
        plt.legend()
        plt.ylim(0,1.5)
        plt.xlim(0,14)
        plt.title(self.name +'_Prop_Mass_Flow')
        filename = self.name +'_Prop_Mass_Flow.png'
        plt.savefig(os.path.join(self.path,filename), dpi=300)
        plt.show()
        
    def OF_plot(self):
        rho_fuel = 800
        self.OF = self.ox_mass_flow()/self.fuel_mass_flow(rho_fuel)
        self.OF2 = self.ox_mass_flow_c_star()/self.fuel_mass_flow_inj_dp()
        #plt.plot(self.data.loc[:,'Time'],self.OF, label='using flow meter and tank mass')
        plt.plot(self.data.loc[:,'Time'],self.OF2, label='using injector dp and c_star')
        
        plt.grid(which='major',axis='both',linewidth = 0.8)
        plt.minorticks_on()
        plt.grid(which='minor',axis='both',linewidth = 0.2)
        plt.xlabel('Time [s]')
        plt.ylabel('OF')
        plt.legend()
        plt.ylim(0,3)
        plt.xlim(0,14)
        plt.title(self.name +'_OF-ratio')
        filename = self.name +'_OF-ratio.png'
        plt.savefig(os.path.join(self.path,filename), dpi=300)
        plt.show()
        
    def isp_plot(self):
        plt.plot(self.data.loc[:,'Time'],self.isp(), label='using injector dp and c_star')
        
        plt.grid(which='major',axis='both',linewidth = 0.8)
        plt.minorticks_on()
        plt.grid(which='minor',axis='both',linewidth = 0.2)
        plt.xlabel('Time [s]')
        plt.ylabel('Specific Impulse [s]')
        plt.legend()
        plt.ylim(0,250)
        plt.xlim(0,14)
        plt.title(self.name +'_Specific-Impulse')
        filename = self.name +'_Isp.png'
        plt.savefig(os.path.join(self.path,filename), dpi=300)
        plt.show()
    
    def Cd_A_test_calc(self, rho_fuel):
        
        Fuel_Inj.Cd_A_calc(self, 'P_Fuel_Injector', 'P_Chamber', rho_fuel=rho_fuel)
        Channels.Cd_A_calc(self, 'Fuel Channel Inlet', 'P_Fuel_Injector', rho_fuel=rho_fuel)
        Fuel_Valve_150.Cd_A_calc(self, 'P_Fuel_Tank', 'Fuel Channel Inlet', rho_fuel=rho_fuel)
        
        Fuel_Inj.print_Cd_A()
        Channels.print_Cd_A()
        Fuel_Valve_150.print_Cd_A()

        Fuel_Inj.pressure_vs_mass_flow_plot()
        #Channels.pressure_vs_mass_flow_plot()
        Fuel_Valve_150.pressure_vs_mass_flow_plot()
        plt.title(self.name + ' Cd*A Plot')
        plt.grid(which='major',axis='both',linewidth = 0.8)
        plt.minorticks_on()
        plt.grid(which='minor',axis='both',linewidth = 0.2)
        plt.legend()
        plt.show()
        
        return 
    
# Class to contain methods for all the different flow components
class Flow_Element:
    def __init__(self, detailed_name, plot_color):
        self.name = detailed_name
        self.color = plot_color
    
    def Cd_A_calc(self, data, upstream_label, downstream_label, rho_fuel):
        self.rho_fuel = rho_fuel
        dp = data.data.loc[:,upstream_label] - data.data.loc[:,downstream_label]
        Cd_A = data.fuel_mass_flow(self.rho_fuel)/np.sqrt(2*self.rho_fuel*dp)
        self.Cd_A_mean = np.round(np.mean(Cd_A),6)
        self.Cd_A_std = np.round(np.std(Cd_A),6)
        return self.Cd_A_mean, self.Cd_A_std
    
    def print_Cd_A(self):
        print(self.name, 'C_d * Area = ', self.Cd_A_mean, ' +/- ', self.Cd_A_std)
        
    def pressure_vs_mass_flow_plot(self):
        dp = np.arange(0, 20+0.1, 0.1)
        mdot = self.Cd_A_mean*np.sqrt(2*self.rho_fuel*dp)
        mdot_max = (self.Cd_A_mean + self.Cd_A_std)*np.sqrt(2*self.rho_fuel*dp)
        mdot_min = (self.Cd_A_mean - self.Cd_A_std)*np.sqrt(2*self.rho_fuel*dp)
        label = self.name + ' Cd*A = ' + str(self.Cd_A_mean) + ' +/- ' + str(self.Cd_A_std)
        plt.plot(dp, mdot, label=label, color=self.color)
        plt.plot(dp, mdot_max, color=self.color, linestyle='dashed')
        plt.plot(dp, mdot_min, color=self.color, linestyle='dashed')
        plt.xlabel('Pressure Drop [bar]')
        plt.ylabel('Mass Flow Rate [kg/s]')

#----------------------------------------------------------------------
#------------------- Flow Component Definition ------------------------
#----------------------------------------------------------------------
        
Fuel_Inj = Flow_Element('Fuel Injector', plot_color='orange')
Channels = Flow_Element('Regenerative Cooling Channels', plot_color='blue')
Fuel_Valve_150 = Flow_Element('Fuel Valve 150 deg Open', plot_color='green')


#----------------------------------------------------------------------
#---------------------- Flow Test Definition --------------------------
#----------------------------------------------------------------------

# Define names and other metadata of the test as well as the trim times

'''
THAN_10_02_24_A_HOT_1 = Data('20240210_THANOS-A_HOT-FIRE_1_A_RAW-DATA-BACKEND.csv', 
                        r'C:\\Users\\marti\\OneDrive\\Documents\\ICLR\\Hot Fire Test Data\\20240210_THANOS-A_Hot-Fire_1_A Data',
                        '20240210_THANOS-A_HOT-FIRE_1_A',
                        t_start=142,
                        t_ignition=144.2,
                        t_end=160,
                        source='BACKEND',
                        ch8name='Ox Injector')

THAN_11_02_24_R_COLD_1 = Data('20240211_THANOS-R_COLD-FLOW_1_C-BACKEND.csv', 
                        r'C:\\Users\\marti\\OneDrive\\Documents\\ICLR\\Hot Fire Test Data\\20240211_THANOS-R_Cold-Flow_1_C Data',
                        '20240211_THANOS-R_COLD-FLOW_1_C',
                        t_start=2423.5,
                        t_ignition=2423,
                        t_end=2429,
                        source='BACKEND',
                        ch8name='Fuel Channel Inlet')
        
THAN_11_02_24_R_COLD_2 = Data('20240211_THANOS-R_COLD-FLOW_2_A_RAW-DATA-BACKEND.csv', 
                        r'C:\\Users\\marti\\OneDrive\\Documents\\ICLR\\Hot Fire Test Data\\20240211_THANOS-R_Cold-Flow_2_A Data',
                        '20240211_THANOS-R_COLD-FLOW_2_A',
                        t_start=574.5,
                        t_ignition=574.1,
                        t_end=577.5,
                        source='BACKEND',
                        ch8name='Fuel Channel Inlet')

THAN_11_02_24_R_HOT_1 = Data('20240211_THANOS-R_HOT-FIRE_1_A_RAW-DATA-SD-CARD.csv', 
                        r'C:\\Users\\marti\\OneDrive\\Documents\\ICLR\\Hot Fire Test Data\\20240211_THANOS-R_Hot-Fire_1_A Data',
                        '20240211_THANOS-R_HOT-FIRE_1_A',
                        t_start=1024,
                        t_ignition=1024.3,
                        t_end=1028,
                        source='SD-CARD',
                        ch8name='Fuel Channel Inlet')


THAN_24_02_24_R_COLD_1 = Data('20240224_THANOS-R_COLD-FLOW_1_A_RAW-DATA-BACKEND.csv', 
                        r'C:\\Users\\marti\\OneDrive\\Documents\\ICLR\\Hot Fire Test Data\\20240224_THANOS-R_Cold-Flow_1_A Data',
                        '20240224_THANOS-R_COLD-FLOW_1_A',
                        t_start=7435,
                        t_ignition=7439.5,
                        t_end=7455,
                        source='BACKEND',
                        ch8name='Post Regulator')


THAN_24_02_24_R_HOT_1 = Data('20240224_THANOS-R_HOT-FIRE_1_A_RAW-DATA-BACKEND.csv', 
                        r'C:\\Users\\marti\\OneDrive\\Documents\\ICLR\\Hot Fire Test Data\\20240224_THANOS-R_Hot-Fire_1_A Data',
                        '20240224_THANOS-R_HOT-FIRE_1_A',
                        t_start=3611.2,
                        t_ignition=3610.8,
                        t_end=3621.5,
                        source='BACKEND',
                        ch8name='Post Regulator')

THAN_24_02_24_R_HOT_2 = Data('20240224_THANOS-R_HOT-FIRE_2_A_RAW-DATA-SD-CARD.csv', 
                        r'C:\\Users\\marti\\OneDrive\\Documents\\ICLR\\Hot Fire Test Data\\20240224_THANOS-R_Hot-Fire_2_A Data',
                        '20240224_THANOS-R_HOT-FIRE_2_A',
                        t_start=3375.2,
                        t_ignition=3374.6,
                        t_end=3388,
                        source='SD-CARD',
                        ch8name='Post Regulator')


THAN_25_02_24_R_HOT_1 = Data('20240225_THANOS-R_HOT-FIRE_1_A_BACKEND.csv', 
                        '20240225_THANOS-R_Hot-Fire_1_A Data',
                        '20240225_THANOS-R_HOT-FIRE_1_A',
                        t_start=1635.4,
                        t_ignition=1636,
                        t_end=1649,
                        source='BACKEND',
                        ch8name='Post Regulator')

THAN_25_02_24_R_HOT_2 = Data('20240225_THANOS-R_HOT-FIRE_2_A_BACKEND_Throttling.csv', 
                        '20240225_THANOS-R_Hot-Fire_2_A Data',
                        '20240225_THANOS-R_HOT-FIRE_2_A',
                        t_start=3458,
                        t_ignition=3460.2,
                        t_end=3475,
                        source='BACKEND',
                        ch8name='Post Regulator')

EREG_22_03_24_COLD_1 = Data('20240322_EREG_Cold-Flow_1_A Data.csv', 
                        '20240322_EREG_Cold-Flow_1_A Data',
                        '20240322_EREG_Cold-Flow_1_A',
                        t_start=58,
                        t_ignition=59,
                        t_end=68,
                        source='BACKEND',
                        ch8name='Null')

THAN_24_03_24_R_HOT_1 = Data('sen0_telemetry.csv', 
                        r'C:\\Users\\marti\\OneDrive\\Documents\\ICLR\\Hot Fire Test Data\\20240324_THANOS-R_Hot-Fire_1_A Data',
                        '20240324_THANOS-R_HOT-FIRE_1_A',
                        t_start=0,
                        t_ignition=0,
                        t_end=20000,
                        source='BACKEND',
                        ch8name='Ox Injector')

'''
THAN_24_03_24_R_HOT_2 = Data('20240324_THANOS-R_HOT-FIRE_2_A_BACKEND.csv', 
                        '20240324_THANOS-R_Hot-Fire_2_A Data',
                        '20240324_THANOS-R_HOT-FIRE_2_A',
                        t_start=1693,
                        t_ignition=1694.3,
                        t_end=1707,
                        source='BACKEND',
                        ch8name='Ox Injector')

THAN_24_03_24_R_HOT_3 = Data('20240324_THANOS-R_HOT-FIRE_3_A_BACKEND.csv', 
                        '20240324_THANOS-R_Hot-Fire_3_A Data',
                        '20240324_THANOS-R_HOT-FIRE_3_A',
                        t_start=5990,
                        t_ignition=5992.8,
                        t_end=6008,
                        source='BACKEND',
                        ch8name='Ox Injector')

THAN_24_03_24_R_HOT_4 = Data('20240324_THANOS-R_HOT-FIRE_4_A_BACKEND.csv', 
                        '20240324_THANOS-R_Hot-Fire_4_A Data',
                        '20240324_THANOS-R_HOT-FIRE_4_A',
                        t_start=7600,#7854
                        t_ignition=7856.15,
                        t_end=8000,
                        source='BACKEND',
                        ch8name='Ox Injector')

EREG_11_04_24_COLD_1 = Data('20240411_EREG_Cold-Flow_1_A Data.csv', 
                        '20240411_EREG_Cold-Flow_1_A Data',
                        '20240411_EREG_Cold-Flow_1_A',
                        t_start=1684,
                        t_ignition=1685.8,
                        t_end=1690,
                        source='BACKEND',
                        ch8name='Null')

EREG_11_04_24_COLD_2 = Data('20240411_EREG_Cold-Flow_2_A Data.csv', 
                        '20240411_EREG_Cold-Flow_2_A Data',
                        '20240411_EREG_Cold-Flow_2_A',
                        t_start=5418,
                        t_ignition=0,
                        t_end=5430,
                        source='BACKEND',
                        ch8name='Null')

EREG_11_04_24_COLD_3 = Data('20240411_EREG_Cold-Flow_3_A Data.csv', 
                        '20240411_EREG_Cold-Flow_3_A Data',
                        '20240411_EREG_Cold-Flow_3_A',
                        t_start=218,
                        t_ignition=220.5,
                        t_end=235,
                        source='BACKEND',
                        ch8name='Null')

EREG_11_04_24_COLD_4 = Data('20240411_EREG_Cold-Flow_4_A Data.csv', 
                        '20240411_EREG_Cold-Flow_4_A Data',
                        '20240411_EREG_Cold-Flow_4_A',
                        t_start=1662,
                        t_ignition=1664,
                        t_end=1680,
                        source='BACKEND',
                        ch8name='Null')

#----------------------------------------------------------------------
#------------------- Data Processing Operations -----------------------
#----------------------------------------------------------------------

# Call methods from the data class for plotting, processing etc.

#THAN_11_02_24_R_COLD_2.Cd_A_test_calc(rho_fuel=900)

#THAN_10_02_24_A_HOT_1.fuel_flow_vs_pressure()

#THAN_10_02_24_A_HOT_1.low_pressure_plot()

#THAN_11_02_24_R_COLD_1.low_pressure_plot()

#THAN_11_02_24_R_COLD_2.low_pressure_plot()

#THAN_11_02_24_R_HOT_1.thrust_curve_plot()

#THAN_24_02_24_R_HOT_2.low_pressure_plot()

#THAN_24_02_24_R_HOT_2.fuel_flow_vs_pressure()

#THAN_24_02_24_R_HOT_2.thrust_curve_plot()

#THAN_24_02_24_R_HOT_2.temperature_plot()

#THAN_24_02_24_R_HOT_2.impulse()




'''
#plt.scatter(THAN_11_02_24_R_HOT_1.data.loc[:,'P_Chamber'],THAN_11_02_24_R_HOT_1.smoothed_thrust(), label = THAN_11_02_24_R_HOT_1.name, marker = 'o', s=1)
plt.scatter(THAN_24_02_24_R_HOT_1.data.loc[:,'P_Chamber'],0.224809*THAN_24_02_24_R_HOT_1.smoothed_thrust(), label = THAN_24_02_24_R_HOT_1.name, marker = 'o', s=1)
plt.scatter(THAN_24_02_24_R_HOT_2.data.loc[:,'P_Chamber'],0.224809*THAN_24_02_24_R_HOT_2.smoothed_thrust(), label = THAN_24_02_24_R_HOT_2.name, marker = 'o', s=1)
plt.scatter(THAN_25_02_24_R_HOT_1.data.loc[:,'P_Chamber'],0.224809*THAN_25_02_24_R_HOT_1.smoothed_thrust(), label = THAN_25_02_24_R_HOT_1.name, marker = 'o', s=1)

#combined_pressures = pd.concat([THAN_24_02_24_R_HOT_2.data.loc[:,'P_Chamber'], THAN_24_02_24_R_HOT_1.data.loc[:,'P_Chamber']], axis=0)
#combined_thrusts = np.concatenate((np.array(THAN_24_02_24_R_HOT_2.smoothed_thrust()), np.array(THAN_24_02_24_R_HOT_1.smoothed_thrust())), axis=0)

D_throat = 42.5e-3
A_throat = (np.pi*D_throat**2)/4
#CF_A, b = np.polyfit(combined_pressures, combined_thrusts, deg=1)
CF = 1.23
CF_A = A_throat*CF
#b = -1/CF_A
b = -5
CF_Label = 'C_F = ' + str(np.round(CF, 3))

combined_pressures = np.arange(10,20,0.1)
plt.plot(combined_pressures, 0.224809*combined_pressures*1e5*CF_A+b, color = 'red', label=CF_Label)

plt.grid(which='major',axis='both',linewidth = 0.8)
plt.minorticks_on()
plt.grid(which='minor',axis='both',linewidth = 0.2)
plt.xlabel('Chamber Pressure [bar]')
plt.ylabel('Thrust [lbf]')
plt.legend()
plt.title('Thrust vs Chamber Pressure')
plt.savefig('THANOS-R Thrust vs Chamber Pressure', dpi=300)
plt.show()
'''

#THAN_24_02_24_R_HOT_2.CF_impulse(CF_A)

#THAN_25_02_24_R_HOT_1.CF_thrust_curve_plot_lbf(CF_A)

#THAN_25_02_24_R_HOT_1.low_pressure_plot()

#THAN_25_02_24_R_HOT_1.fuel_flow_vs_pressure()

#THAN_25_02_24_R_HOT_1.data.loc[:,'Thrust'] = THAN_25_02_24_R_HOT_1.smoothed_thrust()+85

#THAN_25_02_24_R_HOT_1.data.to_csv(os.path.join(THAN_25_02_24_R_HOT_1.path,THAN_25_02_24_R_HOT_1.name + '_Trimmed_Data.csv'))

#THAN_24_02_24_R_HOT_1.temperature_plot()
#THAN_24_02_24_R_HOT_2.temperature_plot()
#THAN_25_02_24_R_HOT_1.temperature_plot()

EREG_11_04_24_COLD_4.temperature_plot(['N2 Upstream'])

EREG_11_04_24_COLD_4.high_pressure_plot()

EREG_11_04_24_COLD_4.low_pressure_plot()

EREG_11_04_24_COLD_4.fuel_flow_vs_pressure()

#THAN_25_02_24_R_HOT_2.thrust_curve_plot()

#THAN_24_03_24_R_HOT_4.thrust_curve_plot_CPLC(0.5,12)

#THAN_24_03_24_R_HOT_4.low_pressure_plot()

#THAN_24_03_24_R_HOT_4.fuel_flow_vs_pressure()

#THAN_24_03_24_R_HOT_4.temperature_plot(('Fuel Pre Channels','Fuel Post Channels','Fuel Post Channels'))

#THAN_24_03_24_R_HOT_4.ox_mass_plot(-11.2)


#THAN_24_03_24_R_HOT_2.m_dot_prop_plot()
#THAN_24_03_24_R_HOT_3.m_dot_prop_plot()
#THAN_24_03_24_R_HOT_4.m_dot_prop_plot()

#THAN_24_03_24_R_HOT_2.OF_plot()
#THAN_24_03_24_R_HOT_3.OF_plot()
#THAN_24_03_24_R_HOT_4.OF_plot()

#THAN_24_03_24_R_HOT_2.impulse()
#THAN_24_03_24_R_HOT_3.impulse()
#THAN_24_03_24_R_HOT_4.impulse()

#THAN_25_02_24_R_HOT_2.animated_thrust_trace()

#THAN_24_03_24_R_HOT_2.isp_plot()
#THAN_24_03_24_R_HOT_2.m_dot_prop_plot()
#THAN_24_03_24_R_HOT_2.OF_plot()
#THAN_24_03_24_R_HOT_2.ox_mass_plot(-11.2)
#THAN_24_03_24_R_HOT_2.ox_mass_from_c_star()


#THAN_24_03_24_R_HOT_4.temperature_plot(['Fuel Pre Channels','Nitrous Tank','Fuel Post Channels'])