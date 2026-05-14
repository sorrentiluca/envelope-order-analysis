from .config import Config, Segment, SegmentResult, AggResult, Candidate, Results
from .io import load_data, load_many, check_columns, extract_channels, estimate_fs, report_encoder_resolution
from .segmentation import (
    unwrap_angle, sort_dedup_angle, find_event_windows,
    find_plateau_in_window, find_constant_speed_segments,
    pick_segments, concatenate_matched_segments, extract_segment_signal,
)
from .resampling import antialias_lowpass, resample_to_uniform_angle
from .spectrum import (
    compute_order_spectrum, interpolate_to_common_orders,
    average_spectra_rms, time_synchronous_average, residual_after_tsa,
)
from .envelope import (
    bandpass_filter, hilbert_envelope, fast_kurtogram,
    auto_select_band, envelope_order_spectrum,
)
from .detectors import (
    fit_noise_floor, snr_spectrum, harmonic_sum_score,
    real_cepstrum, detect_fundamentals,
)
from .campbell import compute_campbell
from .plotting import (
    AXIS_COLORS, bpf_reference_orders, add_bpf_reference_lines,
    annotate_box, add_bpf_vlines,
    plot_raw_and_angle_waveform, plot_order_spectrum,
    plot_envelope_order_spectrum, plot_kurtogram,
    plot_campbell, plot_detector_score,
)
