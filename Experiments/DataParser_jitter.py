import argparse
import glob
import numpy as np
import scipy as sp
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os

from ExperimentManager import estimateSNR, estimateNoiseFloor, plotPowerSpectralDensity


"""
 N_fft = (1 << 16)

            freq, inputSpec = signal.welch(u_hat, 1./Ts, axis=0, nperseg = N_fft, scaling='density')


            if N_fft > 2*len(freq):     # Make sure the FFT length is consistent
                N_fft = 2*len(freq)

            # fig, ax = plt.subplots(nrows=1, ncols=1)
            tf_abs = np.abs(inputSpec[:,0])
            # ax.semilogx(freq, 10 * np.log10(tf_abs))
            # ax.grid()

            

            
            if jitterAmount:
                title = "SNR - Jitter: {}".format(jitterAmount)
            else:
                title = "SNR"
            plt.figure()
            plt.title(title)
            plt.grid()
            # spectralDensity, sinusoidFrequency, N_FFT, Ts, f, num_signal_bins = 5, signalBand=None, SNR=None
            
            snr_est = estimateSNR(tf_abs, frequency, N_fft, Ts, f=freq)
            plotPowerSpectralDensity(tf_abs, frequency, N_fft, Ts, freq, SNR=snr_est)


            # plt.figure()
            # plt.title("SNR Extra Input")
            # plt.grid()
            # snr_est_extra_inp = estimateSNR(tf_abs_extra_inp, frequency, N_fft, Ts, f=freq)
            # print("SNR: {}\nSNR extra input: {}".format(snr_est, snr_est_extra_inp))

            
            # noise_floor = estimateNoiseFloor(tf_abs, frequency, N_fft, Ts)
            # noise_floor_extra_inp = estimateNoiseFloor(tf_abs_extra_inp, frequency, N_fft, Ts)
            # print("Noise Floor: {}\nNoise Floor extra input: {}".format(noise_floor, noise_floor_extra_inp))
            

            plt.show()
"""

def plotStateTrajectory(simulation_file_path, states):
    with open(simulation_file_path, 'rb') as simulation_file:
        result = pkl.load(simulation_file)

    plt.plot(result['t'], result['output'][:,states])
    plt.show()



def main(**kwargs):
    # spectralDensity, sinusoidFrequency, N_FFT, Ts, f, num_signal_bins = 5, signalBand=None, SNR=None
    data_dir = kwargs['data_dir']
    u_hat_files = glob.glob(os.path.join(data_dir, "reconstructions", "u_hat*.csv"))
    cwd = os.getcwd()
    with open(os.path.join(data_dir, "recon_required.pkl"), 'rb') as recon_files:
        t,sys,inputs = pkl.load(recon_files)

    compare_u_hat_files = glob.glob(os.path.join(cwd, "simulation_data", "80Hz_noise1e-09_jitter0.01_JitterEstimation_5_States","reconstructions", "u_hat*.csv"))
    
    first_file = u_hat_files[0]
    compare_file = compare_u_hat_files[0]

    first = pd.read_csv(first_file).values

    uhat_jitter = first[:,1]
    first = first[:,0]
    compare = pd.read_csv(compare_file).values

    Ts = 80e-6
    N_fft = (1<<12)

    freq, jitter_spec = signal.welch(uhat_jitter,
                                   1./Ts,
                                   axis=0,
                                   nperseg = N_fft,
                                   nfft = (1<<16),
                                   scaling='density')
    plt.semilogx(freq,10*np.log10(jitter_spec), label="Gaussian Jitter PSD")
    plt.legend()
    plt.show()

    freq, first_spec = signal.welch(first,
                                   1./Ts,
                                   axis=0,
                                   nperseg = N_fft,
                                   nfft = (1<<16),
                                   scaling='density')

    freq, compare_spec = signal.welch(compare,
                                   1./Ts,
                                   axis=0,
                                   nperseg = N_fft,
                                   nfft = (1<<16),
                                   scaling='density')
    
    first_spec_abs = np.abs(first_spec)
    compare_spec_abs = np.abs(compare_spec)

    plt.semilogx(freq[10:],10*np.log10(first_spec_abs[10:]), label="Should be better")
    plt.semilogx(freq[10:],10*np.log10(compare_spec_abs[10:]), label="Should be worse")
    plt.legend()
    plt.show()
    
    # N_fft = 39999
    # sinusoidFrequency, N_FFT, Ts, f=None, num_signal_bins = 5, signalBand=None):
    first_SNR = estimateSNR(first_spec_abs, 80, N_fft, Ts)
    compare_SNR = estimateSNR(compare_spec_abs, 80, N_fft, Ts)
    
    print("First SNR: {}".format(first_SNR))
    print("Compare SNR: {}".format(compare_SNR))





    if False:
        data_dir = kwargs['data_dir']
        u_hat_files = glob.glob(os.path.join(data_dir, "reconstructions", "u_hat*.csv"))
        simulation_files = glob.glob(os.path.join(data_dir, "simulation*.pkl"))
        plotStateTrajectory(simulation_files[0], states=(0,1,2,3,4))

        specifications = pd.read_csv(os.path.join(data_dir, "specifications.csv"))

        psd = dict()
        SNR_NF = dict()
        u_hat_dict = dict()
        for u_hat_path in u_hat_files:
            u_hat = pd.read_csv(u_hat_path).values
            
            Ts = 80e-6
            t = np.linspace(0., (len(u_hat)-1)*Ts, len(u_hat))
            N_fft = (1 << 16)
            freq, inputSpec = signal.welch(u_hat,
                                           1./Ts,
                                           axis=0,
                                           nperseg = N_fft,
                                           scaling='density')
            if N_fft > 2*len(freq):     # Make sure the FFT length is consistent
                N_fft = 2*len(freq)

            tf_abs = np.abs(inputSpec[:, 0])
            noise_floor_estimate = estimateNoiseFloor(tf_abs, 70, N_fft, Ts)

            
            tmp = u_hat_path.split("/")[-1].split("_")[-1][:-4]
            key = float(tmp)
            psd[key] = 10*np.log10(tf_abs)
            u_hat_dict[key] = u_hat.flatten()

            snr_est = estimateSNR(tf_abs, 70, N_fft, Ts, f=freq)
            SNR_NF[key] = {'SNR':snr_est, 'Noise Floor': noise_floor_estimate}

            
        psd['frequency'] = freq
        u_hat_dict['time'] = t
        
        tmpidx = data_dir.split("/")[-1].find('jitter')
        jitterAmount = data_dir.split("_")[-1][6:]

        psd_df = pd.DataFrame(psd).set_index('frequency')
        psd_df_keys = psd_df.keys().sort_values(ascending=False)
        ax = psd_df.loc[10:,psd_df_keys].plot(logx=True,
                                              subplots=True, 
                                              grid=True,
                                              sharey=True,
                                              title="PSD of $\hat{u}$ - Simulation with Jitter = %sTs" % jitterAmount)

        props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
        for idx in range(len(ax)):
            textstr ='\n'.join((r'SNR $\approx {:.8e}$'.format(SNR_NF[psd_df_keys[idx]]['SNR']),
                                r'Noise Floor $\approx {:.8e}$'.format(SNR_NF[psd_df_keys[idx]]['Noise Floor']),
                                r'$\sigma_{{jitter}} = {:.1e}$'.format(psd_df_keys[idx])))

            ax[idx].text(2.5,-100, textstr, fontsize=10, bbox=props)

        u_hat_df = pd.DataFrame(u_hat_dict).set_index('time')
        u_hat_df_keys = u_hat_df.keys().sort_values(ascending=False)
        ax1 = u_hat_df.loc[:,psd_df_keys].plot(subplots=True, 
                                              grid=True,
                                              sharey=True,
                                              title="$\hat{u}$ - Simulation with Jitter = %sTs" % jitterAmount)
        for idx in range(len(ax1)):
            textstr_uhat = (r'$\sigma_{jitter} = %.5f$' % (u_hat_df_keys[idx]))

            ax1[idx].text(1,1, textstr_uhat, fontsize=10, bbox=props)

        plt.show()
        SNR_df = pd.DataFrame(SNR_NF)
    

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="ADC Project\
                                         experiment manager")
    arg_parser.add_argument("--data_dir", required=True, type=str)

    args = vars(arg_parser.parse_args())
    main(**args)
