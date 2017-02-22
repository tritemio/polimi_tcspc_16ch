"""
Utility functions to work with TCSPC nanotimes decays.

License: MIT
Copyright 2017 Antonino Ingargiola <tritemio@gmail.com>
"""

import numpy as np


def fullwidth_alphamax(x, y_peak, alpha):
    """Compute full-width at fraction `alpha` of max of passed curve.

    Computes the curve full-width at given percentage of the max.
    To compute the FWHM use `alpha = 0.5`.

    Arguments:
        x, y_peak (arrays): the peak coordinates. It must have a monotonic
            rise and a monotonic fall.
        alpha (float): fraction of the peak max at which the full width
            is computed. Valid values are in the range (0, 1).

    Returns:
        Tuple of two floats: full-width of the peak and x-axis position where
        the raising edge reaches a fraction `alpha` of its max.
    """
    y_unit = y_peak / y_peak.max()
    t_rise = np.interp(alpha,
                       xp=y_unit[:y_unit.argmax()],
                       fp=x[:y_unit.argmax()])
    idx_fall_stop = np.where(y_unit[y_unit.argmax():] < alpha)[0][0]
    xp = y_unit[y_unit.argmax():y_unit.argmax() + idx_fall_stop + 1][::-1]
    fp = x[y_unit.argmax():y_unit.argmax() + idx_fall_stop + 1][::-1]
    assert (np.diff(xp) >= 0).all()
    t_fall = np.interp(alpha, xp=xp, fp=fp)
    fullwidth = t_fall - t_rise
    return fullwidth, t_rise


def calc_nanotime_hist(nanotimes, unit, idxmax, idxmin=0, rebin=1,
                       t_start=-np.inf, t_stop=np.inf, offset=0, pdf=False):
    """Compute nanotimes histogram.

    Arguments `t_start`, `t_stop`, `offset` and `unit` need to use in the
    same "units", for example nanoseconds.

    Arguments:
        idxmin, idxmax (int): min and max bin range in raw nanotimes units.
        rebin (int): bin width in raw nanotimes units.
        offset (float): time (in `unit`) to use as time=0 in the histogram
            time axis.
        pdf (bool): if True normalize the histogram to be a PDF.
        t_start, t_stop (float): min/max time range (in `unit`) for the
            returned histogram. The range selection is applied after
            translating nanotimes by `offset`.

    Returns:
        Two arrays: bin centers in `units` and histogram of nanotimes.
    """
    bins_raw = np.arange(idxmin, idxmax, rebin)

    nanot_hist, _ = np.histogram(nanotimes, bins=bins_raw)
    bin_centers = ((bins_raw[:-1] + 0.5 * (bins_raw[1] - bins_raw[0])) * unit -
                   offset)
    if t_start > -np.inf or t_stop < np.inf:
        mask = (bin_centers >= t_start) * (bin_centers <= t_stop)
        bin_centers = bin_centers[mask]
        nanot_hist = nanot_hist[mask]
    if pdf:
        nanot_hist = (nanot_hist.astype('float') /
                      (nanot_hist.sum() * unit * rebin))
    return bin_centers, nanot_hist


def calc_nanotime_hist_ich_ns(d, ich=0, idxmax=None, idxmin=0, inbursts=False,
                              rebin=1, t_start=-np.inf, t_stop=np.inf,
                              offset=0, pdf=False):
    """Compute nanotimes histogram from spot `ich` in the `Data` object `d`.

    Arguments `t_start`, `t_stop`, `offset` are in nanoseconds.

    Arguments:
        d (Data object): object containing the nanotimes.
        ich (int): channel number. Default 0.
        idxmin, idxmax (int): min and max bin range in raw nanotimes units
            used to compute the histogram. If not specified, uses the full
            range of nanotimes.
        rebin (int): bin width in raw nanotimes units.
        offset (float): time in nanoseconds to use as time=0 in the histogram
            time axis.
        pdf (bool): if True normalize the histogram to unit area.
        t_start, t_stop (float): min/max time range in nanoseconds for the
            returned histogram. The range selection is applied after
            translating nanotimes by `offset`.

    Returns:
        Two arrays: bin centers in `units` and histogram of nanotimes.
    """
    unit = d.nanotimes_params[ich]['tcspc_unit'] * 1e9
    if idxmax is None:
        idxmax = d.nanotimes_params[ich]['tcspc_num_bins']
    ntimes = d.nanotimes[ich]
    if inbursts:
        ntimes = ntimes[d.ph_in_bursts_mask_ich(ich)]
    bin_centers, nanot_hist = calc_nanotime_hist(
        ntimes, unit, idxmax=idxmax, idxmin=idxmin, rebin=rebin,
        t_start=t_start, t_stop=t_stop, offset=offset, pdf=pdf)
    return bin_centers, nanot_hist


def decay_hist_offset(nanotimes, unit, idxmax, idxmin=0, rmin=0.1, rmax=0.9,
                      cross_th=0.5, rebin=1, pdf=True):
    """Compute the offset of the decay.

    Compute the offset of the decay as the time point where the rising edge
    crosses a threshold `cross_th` as percentage its max (by default 50% of
    the max).

    Arguments:
        nanotimes (array of ints): array of raw nanotimes
        unit (float): nanotimes unit
        idxmax (int): max raw nanotime value included in the histogram.
        idxmin (int): min raw nanotime value included in the histogram.
            Default 0.
        rmin, rmax (float): fraction of the max where the monotonic
            rise starts (`vmin`) or ends (`vmax`). Valid values are in the
            (0, 1) range.
        cross_th (float): fraction of the max used as threshold to compute
            the crossing time. The returned value is the time where the
            rising edge crosses this threshold.
        rebin (int): rebinning of nanotimes histogram. Default 1 (no rebin).
        pdf (bool): if True normalize the histogram to unit area.
    """
    assert 0 < rmin < cross_th < rmax < 1
    t, hist = calc_nanotime_hist(nanotimes, unit, idxmin=idxmin,
                                 idxmax=idxmax, rebin=rebin, pdf=pdf)
    vcross = hist.max() * cross_th
    vmin, vmax = rmin * hist.max(), rmax * hist.max()

    idxpeak = hist.argmax()
    # last point below the threshold
    idxcross_minus = np.where(hist[:idxpeak] < vcross)[0][-1]
    # first point above the threshold
    idxcross_plus = np.where(hist[:idxpeak] > vcross)[0][0]

    # NOTE: This function may be simply computing t_cross
    # interpolating between t[idxcross_minus] and t[idxcross_plus].
    # However I use a larger range (defined by rmin and rmax)
    # to make sure that the rising-edge is monotonic and therefore
    # the offset computation is accurate.

    # monotonic raising edge start: last point <= vmin
    idxstart = np.where(hist[:idxpeak] <= vmin)[0][-1]
    # monotonic raising edge stop: first point >= vmax
    idxstop = np.where(hist >= vmax)[0][0]

    # The range needs to include the stop point
    ti, yi = t[idxstart:idxstop + 1], hist[idxstart:idxstop + 1]
    assert (np.diff(yi) > 0).all(), 'Raising edge non-monotonic.'
    assert idxstart <= idxcross_minus < idxcross_plus <= idxstop

    t_cross = np.interp(vcross, yi, ti)
    assert t[idxcross_minus] <= t_cross <= t[idxcross_plus]
    return t_cross
