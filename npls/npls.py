import numpy as np
from astropy.stats import LombScargle

def chi2(t, y, dy):
    w = np.power(dy, -2)
    ybar = np.dot(w, y) / sum(w)

    return sum(w * (y - ybar)**2)

def nonparametric_lomb_scargle(t, y, dy,
                               minimum_frequency=1./100.,
                               maximum_frequency=1./0.2,
                               samples_per_peak=5,
                               max_nterms=10):
    """
    Non-parametric multi-harmonic Lomb Scargle periodogram

    Uses the Bayesian Information Criterion to automatically
    choose the best number of harmonics to use at each
    trial frequency.

    Parameters
    ----------
    t: array_like
        Observation times.
    y: array_like
        Observations.
    dy: array_like
        Observation uncertainties.
    minimum_frequency: float
        Minimum frequency to search.
    maximum_frequency: float
        Maximum frequency to search.
    samples_per_peak: float
        Oversampling factor.
    max_nterms: int
        Maximum number of terms to use for any fit.
    """
    chi2_0 = chi2(t, y, dy)
    n = len(t)

    autopower_kwargs = dict(minimum_frequency=minimum_frequency,
                            maximum_frequency=maximum_frequency,
                            samples_per_peak=samples_per_peak)


    # compute mhgls periodograms for each harmonic h=1, 2, ..., H
    nterms = 1 + np.arange(max_nterms)
    periodograms = [(LombScargle(t, y, dy, nterms=h)
                     .autopower(**autopower_kwargs))
                    for h in nterms]

    frequencies = periodograms[0][0]
    periodograms = [p for f, p in periodograms]

    # add bic penalties
    periodograms = [(p * chi2_0 - 2 * h * np.log(n)) / (chi2_0 + np.log(n))
                    for p, h in zip(periodograms, nterms)]

    # get the max over all harmonics
    p_np = np.stack(periodograms).T.max(axis=-1)

    return frequencies, p_np



if __name__ == '__main__':
    import argparse
    import pandas as pd
    import matplotlib.pyplot as plt
    from time import time

    parser = argparse.ArgumentParser(description='Nonparametric Lomb Scargle')
    parser.add_argument('filename', help='Path to lightcurve csv file')

    args = parser.parse_args()

    lc = pd.read_csv(args.filename, names=['t', 'y', 'dy'])

    max_nterms=10
    print("Running npls")
    t0 = time()
    freqs, npls = nonparametric_lomb_scargle(lc.t, lc.y, lc.dy,
                                             max_nterms=max_nterms)
    print("  done in %.3e"%(time() - t0) + " s")

    print("Running ls")
    t0 = time()

    ls = LombScargle(lc.t, lc.y, lc.dy).power(freqs)
    print("  done in %.3e"%(time() - t0) + " s")


    print("Running mhls h=%d"%(max_nterms))
    t0 = time()

    mhls = LombScargle(lc.t, lc.y, lc.dy, nterms=max_nterms).power(freqs)
    print(" done in %.3e s"%(time() - t0))

    f, ax = plt.subplots()
    ax.plot(freqs, ls / ls.max(), color='g', alpha=0.7, lw=0.5, label="Lomb-Scargle")
    ax.plot(freqs, npls / npls.max(), color='r', alpha=0.7, lw=0.5,
            label="NPLS (H=%d)"%(max_nterms))
    ax.plot(freqs, mhls / mhls.max(), color='b', alpha=0.7, lw=0.5,
            label="MHGLS (H=%d)"%(max_nterms))
    ax.axhline(1, ls=':', color='k')
    ax.legend(loc='best', fontsize=9)
    ax.set_xlabel('Frequency [days$^{-1}$]')
    ax.set_ylabel('$P / P_{\\rm max}$')

    plt.show()



