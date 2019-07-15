from .examples import *
import matplotlib.pyplot as plt
import time

__all__ = ['run']

def run():
    print('Running all examples.')
    t0 = time.time()
    res = bragg_reflectivity_sigma()
    res2 = bragg_reflectivity_pi_asymmetric()
    res3 = laue_symmetric()
    res4 = bragg_bent_symmetric()
    print('Total running time: ' + str(time.time()-t0) + ' seconds.')

    plt.figure()
    plt.plot(res[0],res[1],linewidth = 2,label='pyTT')
    plt.plot(res[2],res[3],linewidth = 2,label='XINPRO 1.2\n(T=25 K)')

    plt.title('Unbent GaAs (004) in Bragg geometry')
    plt.xlabel('$\\theta-\\theta_B$ (arc sec)')
    plt.ylabel('Reflectivity')
    plt.legend()

    plt.figure()
    plt.plot(res2[0],res2[1],linewidth = 2,label='pyTT')
    plt.plot(res2[2],res2[3],linewidth = 2,label='XINPRO 1.2\n(T=75 K)')

    plt.title('Unbent LiF (220) with 10 deg asymmetry in Bragg geometry')
    plt.xlabel('$\\theta-\\theta_B$ (arc sec)')
    plt.ylabel('Reflectivity')
    plt.legend()

    plt.figure()
    plt.plot(res3[0],res3[1],linewidth = 2,label='pyTT (reflected)')
    plt.plot(res3[0],res3[2],linewidth = 2,label='pyTT (transmitted)')
    plt.plot(res3[3],res3[4],linewidth = 2,label='XINPRO 1.2\n(reflected, T=50 K)')
    plt.plot(res3[3],res3[5],linewidth = 2,label='XINPRO 1.2\n(transmitted, T=50 K)')

    plt.title('Unbent Ge (111) in Laue geometry')
    plt.xlabel('$\\theta-\\theta_B$ (arc sec)')
    plt.ylabel('Reflectivity')
    plt.legend()

    plt.figure()
    plt.plot(res4[0],res4[1],linewidth = 2,label='pyTT')

    plt.title('Cylindrically bent (R_b = 1 m) Si (660) in Bragg geometry at 88 deg')
    plt.xlabel('$\Delta E$ (meV)')
    plt.ylabel('Reflectivity')
    plt.legend()


    plt.show()
