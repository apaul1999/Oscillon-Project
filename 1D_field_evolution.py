# 1D_field_evolution.py
# Simulates an oscillatory scalar field in 1+1 spacetime with a polynomial potential

import numpy as np
import numba, h5py, os
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF
from PyPDF2 import PdfFileMerger, PdfFileReader

def animate_phi(frame, line, aplot):
    '''
    Animation function for the field amplitude that is passed to the FuncAnimation class

    Parameter(s):
    frame: (int) the frame number
    line: the plot lines array
    aplot: (ndarray, shape=(N,nplots)) 2D numpy array containing the values of phi for different times

    Returns:
    line: the updated plot lines array
    '''
    line.set_ydata(aplot[:,frame])  # Update the data
    return line,


def animate_rho(frame, line, rplot):
    '''
    Animation function for the hamiltonian that is passed to the FuncAnimation class

    Parameter(s):
    frame: (int) the frame number
    line: the plot lines array
    rplot: (ndarray, shape=(N,nplots)) 2D numpy array containing the values of rho at different times

    Returns:
    line: the updated plot lines array
    '''
    line.set_ydata(rplot[:,frame])  # Update the data
    return line,


@numba.jit(nopython=True, parallel=True)
def evolve(L, N, dt, nSteps, w, phi, phi_dot):
    '''
    Computes the evolution of a scalar field in 1 + 1 (d = 1) spacetime with a polynomial potential and periodic boundary conditions 
    using first order central difference schemes in both time and space.

    Parameter(s):
    L: (float) size of the box
    N: (int) number of grid points
    dt: (float) size of the time step
    nSteps: (int) number of time steps
    phi: (ndarray, shape=(N,)) 1D numpy array containing the initial field amplitudes
    phi_dot: (ndarray, shape=(N,)) 1D numpy array containing the initial field velocities
    w: the oscillation angular frequency in physical time units

    Returns:
    phi: (ndarray, shape=(N,)) 1D numpy array containing the field amplitudes
    rho: (ndarray, shape=(N,)) 1D numpy array containing the hamiltonian
    x: (ndarray, shape=(N,)) 1D numpy array containing the x coordinates of the spatial grid
    aplot: (ndarray, shape=(N,nplots)) 2D numpy array containing the values of phi for different times
    tplot: (ndarray, shape=(nplots,)) 1D numpy array containing the time coordinates for plotting
    rplot: (ndarray, shape=(N,nplots)) 2D numpy array containing the values of rho at different times
    eplot: (ndarray, shape=(nplots,)) 1D numpy array containing the energy values at different times
    iplot: (int) last updated plot number
    E_stats: (tuple) Tuple containing the energy statistics of the run (mean, std, rstd, normalized range)
    '''
    # Compute constants
    dr = L/N
    coeff1 = dt**2
    coeff2 = dr**2
    coeff3 = coeff1/coeff2

    # Set the initial conditions and boundary conditions
    x = np.arange(N)*dr - L/2
    ip = np.arange(N) + 1
    im = np.arange(N) - 1
    ip[N-1] = 0    # Periodic BC
    im[0] = N-1

    # Initialize plotting data
    nplots = int(np.around(60*dt*nSteps*w/(2*np.pi)))
    iplot = 1
    aplot = np.empty((N, nplots))
    tplot = np.empty(nplots)
    aplot[:,0] = np.copy(phi)
    tplot[0] = 0
    rplot = np.empty((N, nplots))
    eplot = np.empty(nplots)
    plotStep = nSteps/nplots

    # Calculate the 1st time step
    phi_old = phi[:] - dt*phi_dot[:]    # Obtained from the backwards time difference scheme with the phi_dot initial condition
    phi_new = coeff3*(phi[ip]-2*phi[:]+phi[im]) + 2*phi[:] - phi_old[:] - coeff1*(phi[:]-phi[:]**3+0.4*phi[:]**5)

    rho = 0.5*phi_dot[:]**2 + 0.5*((phi[ip]-phi[im])/(2*dr))**2 + 0.5*phi[:]**2 - 0.25*phi[:]**4 + (1/15)*phi[:]**6
    rplot[:,0] = np.copy(rho)
    mask_array = np.arange(1, N-1, 2)    # Index mask array to calculate the reduced hamiltonian's integral using Simpson's rule
    E = np.sum(((2*dr/3)*(rho[ip]+4*rho[:]+rho[im]))[mask_array])
    eplot[0] = E

    # MAIN LOOP
    for iStep in range(0, nSteps):
        # Compute the new values of the field amplitude
        phi_old = np.copy(phi)
        phi = np.copy(phi_new)
        phi_new[:] = coeff3*(phi[ip]-2*phi[:]+phi[im]) + 2*phi[:] - phi_old[:] - coeff1*(phi[:]-phi[:]**3+0.4*phi[:]**5)
        phi_dot[:] = (phi_new[:]-phi_old[:])/(2*dt)
        
        if (iStep+1) % plotStep < 1:
            # Computes the total energy by integrating over the reduced hamiltonian using Simpson's rule
            rho = 0.5*phi_dot[:]**2 + 0.5*((phi[ip]-phi[im])/(2*dr))**2 + 0.5*phi[:]**2 - 0.25*phi[:]**4 + (1/15)*phi[:]**6
            E = np.sum(((2*dr/3)*(rho[ip]+4*rho[:]+rho[im]))[mask_array])
            
            # Periodically record phi(t), rho(t), and E(t) for plotting
            aplot[:,iplot] = np.copy(phi)
            tplot[iplot] = dt*(iStep+1)
            rplot[:,iplot] = np.copy(rho)
            eplot[iplot] = E
            iplot += 1
    
    # Calculate the Energy Statistics
    E_mean = np.mean(eplot[:iplot])
    E_std = np.std(eplot[:iplot])
    E_rstd = E_std/E_mean
    E_nr = np.ptp(eplot[:iplot])/E_mean
    E_stats = (E_mean, E_std, E_rstd, E_nr)

    return phi, rho, x, aplot, tplot, rplot, eplot, iplot, E_stats


# Select numerical parameters (number of grid points, time step, number of steps, etc.)
L = 50.0    # Size of the box
N = 50000    # Number of grid points
dt = 0.0002    # Size of the time step
nSteps = 5000000    # Number of time steps
w = 0.9    # Oscillation angular frequency in physical time units

# Compute the field evolution
f = h5py.File('1D_oscillon_profile.hdf5', 'r')
phi = f['Dataset1'][:]    # Initial field amplitude
phi_dot = np.zeros(N)    # Initial field velocity
phi, rho, x, aplot, tplot, rplot, eplot, iplot, E_Stats = evolve(L, N, dt, nSteps, w, phi, phi_dot)
interval = 1000*nSteps*dt*w/(2*np.pi*iplot)    # Interval between frames in the animations

# Print and save the run information
input_summary = 'L: %s\nN: %s\ndr: %s\ndt: %s\nnSteps: %s\nw: %s' % (L, N, L/N, dt, nSteps, w)
E_info = 'Energy Mean: %s\nEnergy Std Dev: %s\nEnergy Rel Std Dev: %s\nEnergy Normalized Range: %s' % E_Stats
run_summary = input_summary + '\n' + E_info
print(run_summary)
pdf = FPDF(format='A4')
pdf.add_page()
pdf.set_font('Arial', size = 12)

for line in run_summary.splitlines():
    pdf.cell(200, 10, txt=line, ln=1, align='L')

pdf.output('run_info.pdf')

# Plot the initial and final field amplitudes
fig1 = plt.figure()
plt.title('Initial and Final Field Amplitudes')
plt.xlabel('x')
plt.ylabel('phi(x)')
plt.plot(x, aplot[:,0], '-', x, phi, '-', linewidth=1.0)
plt.legend(['Initial', 'Final'])
#plt.show()

# Plot the initial and final energy densities
fig2 = plt.figure()
plt.title('Initial and Final Hamiltonians')
plt.xlabel('x')
plt.ylabel('rho(x)')
plt.plot(x, rplot[:,0], '-', x, rho, '-', linewidth=1.0)
plt.legend(['Initial', 'Final'])
#plt.show()

# Plot the Energy
fig3 = plt.figure()
plt.title('Energy Conservation vs. Time')
plt.xlabel('t')
plt.ylabel('E(t)')
plt.plot(tplot[:iplot], eplot[:iplot], '-', linewidth=1.0)
#plt.show()

# Save the plots to a PDF
pp = PdfPages('run_plots.pdf')
pp.savefig(fig1)
pp.savefig(fig2)
pp.savefig(fig3)
pp.close()

# Merge the run information and plots PDF's
merger = PdfFileMerger()
for filename in ['run_info.pdf', 'run_plots.pdf']:
    merger.append(filename)

merger.write('run_summary.pdf')
merger.close()
os.remove('run_info.pdf')
os.remove('run_plots.pdf')

# Create an animation of the amplitude evolution
fig4, ax = plt.subplots()
ax.set_title('Time evolution of the Field Amplitude')
ax.set_xlabel('x')
ax.set_ylabel('phi(x)')
ax.set_xlim(-L/2, L/2)
y_min = np.amin(aplot)
y_max = np.amax(aplot)
r = y_max - y_min
ax.set_ylim(y_min-0.05*r, y_max+0.05*r)
line, = ax.plot(x, aplot[:,0], linewidth=1.0)
phi_ani = animation.FuncAnimation(fig4, animate_phi, frames=iplot, fargs=(line, aplot), interval=interval, repeat=True, repeat_delay=500, blit=True)
#plt.show()

# Create an animation of the hamiltonian evolution
fig5, ax = plt.subplots()
ax.set_title('Time evolution of the Hamiltonian')
ax.set_xlabel('x')
ax.set_ylabel('rho(x)')
ax.set_xlim(-L/2, L/2)
y_max = np.amax(rplot)
ax.set_ylim(0, 1.05*y_max)
line, = ax.plot(x, rplot[:,0], linewidth=1.0)
rho_ani = animation.FuncAnimation(fig5, animate_rho, frames=iplot, fargs=(line, rplot), interval=interval, repeat=True, repeat_delay=500, blit=True)
#plt.show()

# Save the animations
writer = animation.FFMpegWriter(fps=1000/interval, bitrate=8000)
print('Saving field amplitude animation')
phi_ani.save('field_amplitude_evolution.mp4', writer=writer, dpi=300)
print('Saving hamiltonian animation')
rho_ani.save('hamiltonian_evolution.mp4', writer=writer, dpi=300)

plt.close('all')

