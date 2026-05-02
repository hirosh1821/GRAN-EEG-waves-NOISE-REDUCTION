import json
from pathlib import Path


def pct_drop(before, after):
    return (before - after) / before * 100.0


def main():
    metrics_path = Path("outputs_v2/metrics_v2.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    print("GRAN EEG Restoration Results")
    print(f"SNR: {metrics['snr_noisy']:.2f} dB -> {metrics['snr_restored']:.2f} dB")
    print(f"SNR improvement: +{metrics['snr_improvement']:.2f} dB")
    print(f"Correlation: {metrics['correlation_noisy']:.3f} -> {metrics['correlation_restored']:.3f}")
    print(f"RMSE: {metrics['rmse_noisy']:.3f} -> {metrics['rmse_restored']:.3f} ({pct_drop(metrics['rmse_noisy'], metrics['rmse_restored']):.1f}% lower)")
    print(f"MAE: {metrics['mae_noisy']:.3f} -> {metrics['mae_restored']:.3f} ({pct_drop(metrics['mae_noisy'], metrics['mae_restored']):.1f}% lower)")
    print(f"Parameters: {int(metrics['n_params']):,}")
    print(f"Training windows: {int(metrics['total_train_windows']):,}")
    print(f"Epochs trained: {int(metrics['epochs_trained'])}")


if __name__ == "__main__":
    main()
