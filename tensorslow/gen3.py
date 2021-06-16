import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


def hh(
    # Parameters
    tmin = 0.0 # ms
    tmax = 50.0 # ms
    # Conductance [mS/cm^2] (Gerstner2014)
    g_Na = 40.0 # Izhikevich2007: 120.0
    g_K = 35.0 # Izhikevich2007: 36.0
    g_L = 0.3 # Izhikevich2007: 0.3
    # Nernst equilibrium potentials [mV] (Gerstner2014)
    E_Na = 55.0 # Izhikevich2007: 120
    E_K = -77.0 # Izhikevich2007: -12
    E_L = -65.0 # Izhikevich2007: 10.6
    # uF/cm^2
    C = 1.0 ):


    '''fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(T, Idv)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(r'Current density (uA/$cm^2$)')
    ax.set_title('Stimulus (Current density)')
    plt.grid()
    plt.show()

    # Neuron potential
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(T, Vy[:, 0])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Vm (mV)')
    ax.set_title('Neuron potential with two spikes')
    plt.grid()
    plt.show()

    # Trajectories with limit cycles
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(Vy[:, 0], Vy[:, 1], label='Vm - n')
    ax.plot(Vy[:, 0], Vy[:, 2], label='Vm - m')
    ax.set_title('Limit cycles')
    ax.legend()
    plt.grid()
    plt.show()'''

'''def test():
    Weights
    w1 = np.random.rand(784, 100)
    w2 = np.random.rand(100, 10)

    for epoch in range(0, epochs):
        # Layer 0 (784, trn_q): Randomly Spiking
        a0 = np.random.rand(784, trn_q) <= a_orig
        # Layer 1 (trn_q, 100): Calc, Normalizing, Randomly Spiking
        a1 = np.dot(a0.T, w1)
        a1 = 0.95 * (a1 / np.max(a1))
        #print_example(a1[:, 0:1])
        a1 = np.random.rand(trn_q, 100) <= a1
        #print_example(a1[:, 0:1])
        # Layer 2 (trn_q, 10): Calc, Normalizing
        a2 = np.dot(a1, w2)
        a2 = 0.95 * (a2 / np.max(a2))

        # STDP
        w1_stdp = np.dot((1*a0), a1) # 784,600 x 600,100
        #print_example(a0[:, 0:1])
        #print_example(a1[:, 0:1])
        w1_stdp = w1_stdp / np.max(w1_stdp)
        #print_example(w1_stdp[:, 0:1])
        w1 = (0.95*w1) + (alpha * w1_stdp * (1 - w1))
        print_example(w1[:, 0:1])
        w2_stdp = np.dot((1*a1.T), soll_trn) #600,100 x 600,10
        #print(soll_trn[0])
        w2_stdp = w2_stdp / np.max(w2_stdp)
        #print_example(w2_stdp[:, 0:1])
        w2 = (0.95*w2) + (alpha * w2_stdp * (1 - w2))

        # Testing the shit
        # Layer 0: Randomly Spiking
        t0 = np.random.rand(784, tst_q) <= t_orig
        # Layer 1: Calc, Normalizing, Randomly Spiking
        t1 = np.dot(t0.T, w1)
        t1 = (t1 - np.min(a1)) / (np.max(t1) - np.min(t1))
        t1 = np.random.rand(tst_q,100) <= t1
        # Layer 2: Calc, Normalizing
        t2 = np.dot(t1, w2)
        t2 = (t2 - np.min(t2))/(np.max(t2)-np.min(t2)) #100,10
        correct = np.sum(np.argmax(t2, axis = 1) == np.argmax(soll_tst, axis = 1))

        print('Epoch:', epoch, int(correct/tst_q * 100),'% correct')
    return'''


# FROM SCARING_ELON.PY

# To Do
# - f-I-curve

class HH_Izhikevich:
    # Shifted Nernst equilibrium potentials [mV] (rested potential V = 0 mV)
    E_K = -12
    E_Na = 120
    E_L = 10.6
    # Conductance [mS/cm^2]
    g_K = 36
    g_Na = 120
    g_L = 0.3
    # Membrane capacitance [uF/cm^2]
    C = 1.0

    def __init__(self):
        self.calc_parameters(0.0)

    def calc_parameters(self, V):
        # Transition rates
        self.alpha_n= (0.01 * (10 - V)) / ((np.exp((10 - V) / 10)) - 1)
        self.beta_n = 0.125 * np.exp(-V / 80)
        self.alpha_m= (0.1  * (25 - V)) / ((np.exp((25 - V) / 10)) - 1)
        self.beta_m = 4 * np.exp(-V / 18)
        self.alpha_h= 0.07 * np.exp(-V / 20)
        self.beta_h = 1 / ((np.exp((30 - V) / 10)) + 1)
        # Steady state
        self.n_inf = self.alpha_n / (self.alpha_n + self.beta_n)
        self.m_inf = self.alpha_m / (self.alpha_m + self.beta_m)
        self.h_inf = self.alpha_h / (self.alpha_h + self.beta_h)
        # Time constant
        self.tau_n = 1 / (self.alpha_n + self.beta_n)
        self.tau_m = 1 / (self.alpha_m + self.beta_m)
        self.tau_h = 1 / (self.alpha_h + self.beta_h)

    def hh_eq(self, y, t0, args):
        V, n, m, h = y
        self.calc_parameters(V)

        I_K = self.g_K * np.power(n, 4) * (V - self.E_K)
        I_Na= self.g_Na* np.power(m, 3) * h * (V - self.E_Na)
        I_L = self.g_L * (V - self.E_L)

        dV = (I(t0) - I_K - I_Na - I_L) / self.C
        # Activation and inactivation dynamics of each channel
        dn = (self.n_inf - n) / self.tau_n
        dm = (self.m_inf - m) / self.tau_m
        dh = (self.h_inf - h) / self.tau_h
        return dV, dn, dm, dh

    def solve(self, T):
        Y = np.array([0.0, n.n_inf, n.m_inf, n.h_inf])
        print(n.n_inf, n.m_inf, n.h_inf)
        sol = odeint(self.hh_eq, Y, T, args=(self, ))
        return sol

class HH_Gerstner:
    # ToDo:
    # Every 2 ms, a random number is drawn from a Gaussian distribution with
    # zero mean and standard deviation σ = 34 μA/cm2. To get a continuous input
    # current, a linear interpolation was used between the target values.

    # Reversal potentials [mV] (fitted by Mainen et al., 1995)
    E_Na= 55
    E_K = -77
    E_L = -65
    # Conductance [mS/cm^2]
    g_Na= 40
    g_K = 35
    g_L = 0.3
    # Membrane capacitance [uF/cm^2]
    C = 1.0

    def __init__(self):
        self.calc_parameters(-65)

    def calc_parameters(self, u):
        # Transition rates
        self.alpha_n = (0.024 * (u - 25)) / (1 - np.exp(-(u - 25) / 9))
        self.beta_n = (-0.002 * (u - 25)) / (1 - np.exp( (u - 25) / 9))
        self.alpha_m = (0.182 * (u + 35)) / (1 - np.exp(-(u + 35) / 9))
        self.beta_m = (-0.124 * (u + 35)) / (1 - np.exp( (u + 35) / 9))
        self.alpha_h = 1 / (1 + np.exp(-(u + 62) / 6))
        self.beta_h = (4 * np.exp((u + 90)/12)) / (1 + np.exp(-(u + 62) / 6))
        #self.alpha_h = (0.024 * (u + 50)) / (1 - np.exp(-(u + 50) / 5))
        #self.beta_h = (-0.002 * (u + 75)) / (1 - np.exp( (u + 75) / 5))


    def hh_eq(self, y, t0, args):
        u, n, m, h = y
        self.calc_parameters(u)

        I_Na= self.g_Na* np.power(m, 3) * h * (u - self.E_Na)
        I_K = self.g_K * np.power(n, 4) * (u - self.E_K)
        I_L = self.g_L * (u - self.E_L)

        du = (I(t0) - I_K - I_Na - I_L) / self.C
        # dm = - (m - m_0(u)) / tau_m(u)
        dn = self.alpha_n * (1 - n) - self.beta_n * n
        dm = self.alpha_m * (1 - m) - self.beta_m * m
        dh = self.alpha_h * (1 - h) - self.beta_h * h
        return du, dn, dm, dh

    def solve(self, T):
        # Asymptotic value
        #self.calc_parameters(-65)
        n_inf = self.alpha_n / (self.alpha_n + self.beta_n)
        m_inf = self.alpha_m / (self.alpha_m + self.beta_m)
        h_inf = self.alpha_h / (self.alpha_h + self.beta_h)
        print(n_inf, m_inf, h_inf)
        Y = np.array([-65, n_inf, m_inf, h_inf])
        sol = odeint(self.hh_eq, Y, T, args=(self, ))
        return sol

class LIF:
    v_rest = -70 # mV
    v_reset = -65 # mV
    firing_threshold = -50 # mV
    #MEMBRANE_RESISTANCE = 10. * b2.Mohm
    #MEMBRANE_TIME_SCALE = 8. * b2.ms
    #ABSOLUTE_REFRACTORY_PERIOD = 2.0 * b2.ms

    def solve(self, T, stim):
        d = T[1]
        x = np.zeros((len(T), 1))
        j = 0
        x[0] = self.v_rest
        #for i in T:
        for i in range(len(T)):
            if i > 0:
                x[j] = x[j-1] + 0.1 * stim[i] * d
                #x[j] = x[j-1] + 0.1 * I(i) * d
                if x[j] > 20:
                    x[j] = self.v_reset
                elif x[j] > self.firing_threshold:
                    x[j] = 20
            j += 1
        return x

def plot(x, y):
    plt.plot(x, y)
    plt.xlabel("T")
    plt.ylabel("y")
    plt.show()

# Stimulus
def I(t):
    if 50.0 < t < 100.0:
        return 5
    elif 150 < t < 250.0:
        return 10
    elif 300 < t < 450:
        return 15
    return 0.0

if __name__ == "__main__":
    n = HH_Izhikevich()
    #n = HH_Gerstner()
    #n = LIF()
    T = np.linspace(0, 500, 5000)

    # Stimulus
    stim = np.zeros((len(T), ))
    stim[500:1000] = 5
    stim[1500:2500] = 10
    stim[3000:4500] = 15

    start = time.time()
    sol = n.solve(T)
    #sol = n.solve(T, stim)
    print(time.time() - start, 'secs')

    plot(T, sol[:, 0])


# FROM MODELS.PY


"""class HhIzhikevich:
    # Shifted Nernst equilibrium potentials [mV] (rested potential V = 0 mV)
    E_K = -12
    E_Na = 120
    E_L = 10.6
    # Conductance [mS/cm^2]
    g_K = 36
    g_Na = 120
    g_L = 0.3
    # Membrane capacitance [uF/cm^2]
    C = 1.0

    def __init__(self):
        self.calc_parameters(0.0)

    def calc_parameters(self, V):
        # Transition rates
        self.alpha_n = (0.01 * (10 - V)) / ((np.exp((10 - V) / 10)) - 1)
        self.beta_n = 0.125 * np.exp(-V / 80)
        self.alpha_m = (0.1  * (25 - V)) / ((np.exp((25 - V) / 10)) - 1)
        self.beta_m = 4 * np.exp(-V / 18)
        self.alpha_h = 0.07 * np.exp(-V / 20)
        self.beta_h = 1 / ((np.exp((30 - V) / 10)) + 1)
        # Steady state
        self.n_inf = self.alpha_n / (self.alpha_n + self.beta_n)
        self.m_inf = self.alpha_m / (self.alpha_m + self.beta_m)
        self.h_inf = self.alpha_h / (self.alpha_h + self.beta_h)
        # Time constant
        self.tau_n = 1 / (self.alpha_n + self.beta_n)
        self.tau_m = 1 / (self.alpha_m + self.beta_m)
        self.tau_h = 1 / (self.alpha_h + self.beta_h)

    def hh_eq(self, y, t0, args):
        V, n, m, h = y
        self.calc_parameters(V)

        I_K = self.g_K * np.power(n, 4) * (V - self.E_K)
        I_Na= self.g_Na* np.power(m, 3) * h * (V - self.E_Na)
        I_L = self.g_L * (V - self.E_L)

        d_V = (I(t0) - I_K - I_Na - I_L) / self.C
        # Activation and inactivation dynamics of each channel
        d_n = (self.n_inf - n) / self.tau_n
        d_m = (self.m_inf - m) / self.tau_m
        d_h = (self.h_inf - h) / self.tau_h
        return d_V, d_n, d_m, d_h

    def solve(self, T):
        Y = np.array([0.0, n.n_inf, n.m_inf, n.h_inf])
        print(n.n_inf, n.m_inf, n.h_inf)
        sol = odeint(self.hh_eq, Y, T, args=(self, ))
        return sol"""

class Lif:
    """Class for Leaky-Integrate-and-Fire-Neurons"""
    C = 1.0

    def __init__(self):
        print("LIF created")

    def diff_eq(self, y_val, x_val):
        """Function for Differential Equation"""
        d_V = 1 # (I(t0) - I_K - I_Na - I_L) / self.C
        return d_V

class TestNeuron:
    """Test"""
    def dy_dx(y, x):
        return x - y

    #xs = np.linspace(0, 5, 100)
    y0 = 1.0  # the initial condition
    #ys = odeint(dy_dx, y0, xs)
    #ys = np.array(ys).flatten()

    # Plot the numerical solution
    #plt.rcParams.update({'font.size': 14})  # increase the font size
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.plot(xs, ys)
    #plt.show()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()

    def animate(i):
        xs = i/100
        ys = odeint(dy_dx, y0, xs)
        ys = np.array(ys).flatten()
        ax.clear()
        #ax.plot(xs, ys)
        ax.plot(xs, ys, "o")
        #ax.plot(car_x+1, car_y+1, "o")
        plt.axis([0, 100, 0, 100])
        #input()

    dummy = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()

def create_neurons():
    """Function for creating neurons"""
    #neuron = Lif()
    neuron = TestNeuron()
    print(type(neuron))

if __name__ == "__main__":
    create_neurons()

"""def create_plot():
    fig = plt.figure(figsize=(8, 8))
    #global ax_track, ax_car, track_x, track_y
    ax_track = fig.add_subplot()
    ax_car = fig.add_subplot()
    #fig, (ax1, ax2) = plt.subplots(sharex=True, sharey=True)

    track_x = [0, 10, 20, 20, 10, 0, 0]
    track_y = [10, 15, 10, 5, 0, 5, 10]

    def animate(i):
        car_x = i
        car_y = i
        ax_track.clear()
        ax_track.plot(track_x, track_y)
        ax_car.plot(car_x, car_y, "o")
        ax_car.plot(car_x+1, car_y+1, "o")
        plt.axis([0, 100, 0, 100])
        #input()

    dummy = animation.FuncAnimation(fig, animate, interval=100)
    #ax1.set_aspect(aspect=1)
    plt.show()"""


#FROM MODELS_OLD.PY


class HhIzhikevich:
    """Hodgkin-Huxley neuron model with shifted rested potential"""
    # Shifted Nernst equilibrium potentials [mV] (rested potential V = 0 mV)
    E_K = -12
    E_Na = 120
    E_L = 10.6
    # Conductance [mS/cm^2]
    g_K = 36
    g_Na = 120
    g_L = 0.3
    # Membrane capacitance [uF/cm^2]
    C = 1.0

    def __init__(self):
        """asd"""
        self.calc_parameters(0.0)

    def calc_parameters(self, V):
        # Transition rates
        self.alpha_n = (0.01 * (10 - V)) / ((np.exp((10 - V) / 10)) - 1)
        self.beta_n = 0.125 * np.exp(-V / 80)
        self.alpha_m = (0.1  * (25 - V)) / ((np.exp((25 - V) / 10)) - 1)
        self.beta_m = 4 * np.exp(-V / 18)
        self.alpha_h = 0.07 * np.exp(-V / 20)
        self.beta_h = 1 / ((np.exp((30 - V) / 10)) + 1)
        # Steady state
        self.n_inf = self.alpha_n / (self.alpha_n + self.beta_n)
        self.m_inf = self.alpha_m / (self.alpha_m + self.beta_m)
        self.h_inf = self.alpha_h / (self.alpha_h + self.beta_h)
        # Time constant
        self.tau_n = 1 / (self.alpha_n + self.beta_n)
        self.tau_m = 1 / (self.alpha_m + self.beta_m)
        self.tau_h = 1 / (self.alpha_h + self.beta_h)

    def hh_eq(self, y, t0, args):
        V, n, m, h = y
        self.calc_parameters(V)

        I_K = self.g_K * np.power(n, 4) * (V - self.E_K)
        I_Na= self.g_Na* np.power(m, 3) * h * (V - self.E_Na)
        I_L = self.g_L * (V - self.E_L)

        d_V = (I(t0) - I_K - I_Na - I_L) / self.C
        # Activation and inactivation dynamics of each channel
        d_n = (self.n_inf - n) / self.tau_n
        d_m = (self.m_inf - m) / self.tau_m
        d_h = (self.h_inf - h) / self.tau_h
        return d_V, d_n, d_m, d_h

    def solve(self, T):
        Y = np.array([0.0, n.n_inf, n.m_inf, n.h_inf])
        print(n.n_inf, n.m_inf, n.h_inf)
        sol = odeint(self.hh_eq, Y, T, args=(self, ))
        return sol

class HhGerstner:
    """Hodgkin-Huxley neuron model"""
    # ToDo:
    # Every 2 ms, a random number is drawn from a Gaussian distribution with
    # zero mean and standard deviation σ = 34 μA/cm2. To get a continuous input
    # current, a linear interpolation was used between the target values.

    # Reversal potentials [mV] (fitted by Mainen et al., 1995)
    E_Na = 55
    E_K = -77
    E_L = -65
    # Conductance [mS/cm^2]
    g_Na = 40
    g_K = 35
    g_L = 0.3
    # Membrane capacitance [uF/cm^2]
    C = 1.0

    def __init__(self):
        self.calc_parameters(-65)

    def calc_parameters(self, u):
        # Transition rates
        self.alpha_n = (0.024 * (u - 25)) / (1 - np.exp(-(u - 25) / 9))
        self.beta_n = (-0.002 * (u - 25)) / (1 - np.exp( (u - 25) / 9))
        self.alpha_m = (0.182 * (u + 35)) / (1 - np.exp(-(u + 35) / 9))
        self.beta_m = (-0.124 * (u + 35)) / (1 - np.exp( (u + 35) / 9))
        self.alpha_h = 1 / (1 + np.exp(-(u + 62) / 6))
        self.beta_h = (4 * np.exp((u + 90)/12)) / (1 + np.exp(-(u + 62) / 6))
        #self.alpha_h = (0.024 * (u + 50)) / (1 - np.exp(-(u + 50) / 5))
        #self.beta_h = (-0.002 * (u + 75)) / (1 - np.exp( (u + 75) / 5))


    def hh_eq(self, y, t0, args):
        u, n, m, h = y
        self.calc_parameters(u)

        I_Na= self.g_Na* np.power(m, 3) * h * (u - self.E_Na)
        I_K = self.g_K * np.power(n, 4) * (u - self.E_K)
        I_L = self.g_L * (u - self.E_L)

        du = (I(t0) - I_K - I_Na - I_L) / self.C
        # dm = - (m - m_0(u)) / tau_m(u)
        dn = self.alpha_n * (1 - n) - self.beta_n * n
        dm = self.alpha_m * (1 - m) - self.beta_m * m
        dh = self.alpha_h * (1 - h) - self.beta_h * h
        return du, dn, dm, dh

    def solve(self, T):
        # Asymptotic value
        #self.calc_parameters(-65)
        n_inf = self.alpha_n / (self.alpha_n + self.beta_n)
        m_inf = self.alpha_m / (self.alpha_m + self.beta_m)
        h_inf = self.alpha_h / (self.alpha_h + self.beta_h)
        print(n_inf, m_inf, h_inf)
        Y = np.array([-65, n_inf, m_inf, h_inf])
        sol = odeint(self.hh_eq, Y, T, args=(self, ))
        return sol

class Lif:
    v_rest = -70 # mV
    v_reset = -65 # mV
    firing_threshold = -50 # mV
    #MEMBRANE_RESISTANCE = 10. * b2.Mohm
    #MEMBRANE_TIME_SCALE = 8. * b2.ms
    #ABSOLUTE_REFRACTORY_PERIOD = 2.0 * b2.ms

    def solve(self, T, stim):
        d = T[1]
        x = np.zeros((len(T), 1))
        j = 0
        x[0] = self.v_rest
        #for i in T:
        for i in range(len(T)):
            if i > 0:
                x[j] = x[j-1] + 0.1 * stim[i] * d
                #x[j] = x[j-1] + 0.1 * I(i) * d
                if x[j] > 20:
                    x[j] = self.v_reset
                elif x[j] > self.firing_threshold:
                    x[j] = 20
            j += 1
        return x

def HH_new():
    # Set random seed (for reproducibility)
    np.random.seed(1000)

    # Start and end time (in milliseconds)
    tmin = 0.0
    tmax = 50.0
    # Average potassium channel conductance per unit area (mS/cm^2)
    gK = 36.0
    # Average sodoum channel conductance per unit area (mS/cm^2)
    gNa = 120.0
    # Average leak channel conductance per unit area (mS/cm^2)
    gL = 0.3
    # Membrane capacitance per unit area (uF/cm^2)
    Cm = 1.0
    # Potassium potential (mV)
    VK = -12.0
    # Sodium potential (mV)
    VNa = 115.0
    # Leak potential (mV)
    Vl = 10.613
    # Time values
    T = np.linspace(tmin, tmax, 10000)

# Potassium ion-channel rate functions
def alpha_n(Vm):
    return (0.01 * (10.0 - Vm)) / (np.exp(1.0 - (0.1 * Vm)) - 1.0)

def beta_n(Vm):
    return 0.125 * np.exp(-Vm / 80.0)

# Sodium ion-channel rate functions
def alpha_m(Vm):
    return (0.1 * (25.0 - Vm)) / (np.exp(2.5 - (0.1 * Vm)) - 1.0)

def beta_m(Vm):
    return 4.0 * np.exp(-Vm / 18.0)

def alpha_h(Vm):
    return 0.07 * np.exp(-Vm / 20.0)

def beta_h(Vm):
    return 1.0 / (np.exp(3.0 - (0.1 * Vm)) + 1.0)

# n, m, and h steady-state values
def n_inf(Vm=0.0):
    return alpha_n(Vm) / (alpha_n(Vm) + beta_n(Vm))

def m_inf(Vm=0.0):
    return alpha_m(Vm) / (alpha_m(Vm) + beta_m(Vm))

def h_inf(Vm=0.0):
    return alpha_h(Vm) / (alpha_h(Vm) + beta_h(Vm))

# Input stimulus
def Id(t):
    if 0.0 < t < 1.0:
        return 150.0
    elif 10.0 < t < 11.0:
        return 50.0
    return 0.0

# Compute derivatives
def compute_derivatives(y, t0):
    dy = np.zeros((4,))

    Vm = y[0]
    n = y[1]
    m = y[2]
    h = y[3]

    # dVm/dt
    GK = (gK / Cm) * np.power(n, 4.0)
    GNa = (gNa / Cm) * np.power(m, 3.0) * h
    GL = gL / Cm

    dy[0] = (Id(t0) / Cm) - (GK * (Vm - VK)) - (GNa * (Vm - VNa)) - (GL * (Vm - Vl))

    # dn/dt
    dy[1] = (alpha_n(Vm) * (1.0 - n)) - (beta_n(Vm) * n)

    # dm/dt
    dy[2] = (alpha_m(Vm) * (1.0 - m)) - (beta_m(Vm) * m)

    # dh/dt
    dy[3] = (alpha_h(Vm) * (1.0 - h)) - (beta_h(Vm) * h)

    return dy

    # State (Vm, n, m, h)
    Y = np.array([0.0, n_inf(), m_inf(), h_inf()])

    # Solve ODE system
    # Vy = (Vm[t0:tmax], n[t0:tmax], m[t0:tmax], h[t0:tmax])
    Vy = odeint(compute_derivatives, Y, T)

    # Input stimulus
    Idv = [Id(t) for t in T]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(T, Idv)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(r'Current density (uA/$cm^2$)')
    ax.set_title('Stimulus (Current density)')
    plt.grid()
    #plt.show()

    # Neuron potential
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(T, Vy[:, 0])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Vm (mV)')
    ax.set_title('Neuron potential with two spikes')
    plt.grid()
    #plt.show()

    # Trajectories with limit cycles
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(Vy[:, 0], Vy[:, 1], label='Vm - n')
    ax.plot(Vy[:, 0], Vy[:, 2], label='Vm - m')
    ax.set_title('Limit cycles')
    ax.legend()
    plt.grid()
    plt.show()

