from __future__ import annotations

import numpy as np
import pytest

import signaltools as st


def tone(sample_rate: int = 2000, n: int = 512) -> list[float]:
    t = np.arange(n) / sample_rate
    return (np.sin(2 * np.pi * 100 * t) + 0.2 * np.sin(2 * np.pi * 400 * t)).tolist()


def test_fir_designs_and_application() -> None:
    x = tone()
    lp = st.fir_lowpass(33, 150, 2000)
    hp = st.fir_highpass(33, 150, 2000)
    bp = st.fir_bandpass(33, 80, 200, 2000)
    bs = st.fir_bandstop(33, 80, 200, 2000)
    fd = st.fractional_delay_fir(33, 0.25)
    diff = st.differentiator_fir()
    assert len(lp.b) == 33
    assert len(st.apply_fir(x, lp)) == len(x)
    assert len(st.apply_fir(x, hp)) == len(x)
    assert len(st.apply_fir(x, bp)) == len(x)
    assert len(st.apply_fir(x, bs)) == len(x)
    assert len(st.apply_fir(x, fd)) == len(x)
    assert len(st.apply_fir(x, diff)) == len(x)


def test_iir_and_biquad_designs() -> None:
    x = tone()
    iir_lp = st.iir_lowpass_single_pole(100, 2000)
    iir_hp = st.iir_highpass_single_pole(100, 2000)
    integ = st.iir_integrator_leaky(0.95)
    bq_lp = st.biquad_lowpass(120, 2000)
    bq_hp = st.biquad_highpass(120, 2000)
    bq_bp = st.biquad_bandpass(120, 2000)
    bq_notch = st.biquad_notch(50, 2000)
    bq_ap = st.biquad_allpass(120, 2000)
    ff_comb = st.comb_filter_feedforward(8, 0.5)
    fb_comb = st.comb_filter_feedback(8, 0.4)
    for filt in [iir_lp, iir_hp, integ, bq_lp, bq_hp, bq_bp, bq_notch, bq_ap, ff_comb, fb_comb]:
        y = st.apply_iir(x, filt)
        assert len(y) == len(x)


def test_savgol_hilbert_and_lms() -> None:
    x = tone()
    coeffs = st.savitzky_golay_coefficients(7, 3)
    filtered = st.savitzky_golay_filter(x, 7, 3)
    hilbert = st.hilbert_transform_fft(x)
    analytic = st.analytic_signal(x)
    env = st.envelope(x)
    desired = x
    reference = [v * 0.8 for v in x]
    lms = st.lms_adaptive_filter(desired, reference, num_taps=4, step_size=0.001)
    assert len(coeffs.b) == 7
    assert len(filtered) == len(x)
    assert len(hilbert) == len(x)
    assert len(analytic) == len(x)
    assert len(env) == len(x)
    assert len(lms.output) == len(x)
    assert len(lms.error) == len(x)
    assert len(lms.weights) == 4


def test_filter_design_validation() -> None:
    with pytest.raises(ValueError):
        st.fir_lowpass(11, 0, 2000)
    with pytest.raises(ValueError):
        st.fir_bandpass(11, 200, 100, 2000)
    with pytest.raises(ValueError):
        st.iir_integrator_leaky(1.5)
    with pytest.raises(ValueError):
        st.savitzky_golay_coefficients(6, 3)
    with pytest.raises(ValueError):
        st.savitzky_golay_coefficients(5, 5)
    with pytest.raises(ValueError):
        st.savitzky_golay_coefficients(5, 3, deriv=4)
    with pytest.raises(ValueError):
        st.apply_iir([1, 2, 3], st.IIRCoefficients(b=[1], a=[0], kind='bad', meta={}))
    empty_lms = st.lms_adaptive_filter([], [], num_taps=4)
    assert empty_lms.weights == [0.0] * 4
