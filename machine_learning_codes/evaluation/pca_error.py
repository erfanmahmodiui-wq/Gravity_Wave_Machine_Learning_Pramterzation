import joblib
import xarray as xr
import numpy as np

from src.config import load_paths


def read_test_data(data_base_path, orography, groupnr):
    Data_Base = joblib.load(data_base_path + "Data_Base_Time_parttion_realscale_group_" + str(groupnr) + ".pkl")

    if orography:
        X_test = Data_Base["X_test_orography"]
        y_test = Data_Base["y_test_orography"]
        ntri   = Data_Base["ntri_orography"]
    else:
        X_test = Data_Base["X_test_nonorography"]
        y_test = Data_Base["y_test_nonorography"]
        ntri   = Data_Base["ntri_nonorography"]

    print(f"max and min of PCA coefficients for group {groupnr} with {'orography' if orography else 'non-orography'}: min={y_test.min():.4e}, max={y_test.max():.4e}, mean={y_test.mean():.4e}  ")
    return y_test, ntri


def reconstruction(EOF_dir, avrage_dir, groupnr, pca_coff, orography, n_90, pca_spectrm_dir, ntri_or):
    if orography:
        fname_EOF  = "EOFs_realscale_orography_" + str(groupnr) + ".nc"
        fname_avg  = "avarage-action-realscale-orography-" + str(groupnr) + ".nc"
        fname_save = "PCA_spectrum_realscale_to_realscale_orography_" + str(groupnr) + ".nc"
    else:
        fname_EOF  = "EOFs_realscale_nonorography_" + str(groupnr) + ".nc"
        fname_avg  = "avarage-action-realscale-nonorography-" + str(groupnr) + ".nc"
        fname_save = "PCA_spectrum_realscale_to_realscale_nonorography_" + str(groupnr) + ".nc"

    with xr.open_dataset(EOF_dir + fname_EOF) as ds:
        EOFs = ds.eigenvectors.data[:, :n_90]
        k = ds.k.data
        l = ds.l.data
        m = ds.m.data

    with xr.open_dataset(avrage_dir + fname_avg) as ds:
        avg = ds.avrage.data.reshape(-1)

    print(f"Average values for group {groupnr} with {'orography' if orography else 'non-orography'}: min={avg.min():.4e}, max={avg.max():.4e}, mean={avg.mean():.4e}")
    reconstructed_EOFs = np.einsum('stc,pc->stp', pca_coff, EOFs)
    reconstructed = reconstructed_EOFs + avg[np.newaxis, np.newaxis, :]
    wave_action_spectrum = reconstructed.reshape(pca_coff.shape[0], pca_coff.shape[1], len(l), len(k), len(m))
    N_action = wave_action_spectrum
    print(f"Reconstructed spectrum range for group {groupnr} with {'orography' if orography else 'non-orography'}: min={wave_action_spectrum.min():.4e}, max={wave_action_spectrum.max():.4e}, mean={wave_action_spectrum.mean():.4e}")

    ds_out = xr.Dataset(
        {"PCA_spectrum": (["ntri", "time", "l", "k", "m"], N_action)},
        coords={"ntri": ntri_or, "time": np.arange(pca_coff.shape[1]), "l": l, "k": k, "m": m}
    )
    ds_out.to_netcdf(pca_spectrm_dir + fname_save, engine="h5netcdf")


def reconstruction_error(real_action_dir, pca_reconstructed_action_dir, data_base_path, groupnr):
    Data_Base = joblib.load(data_base_path + "Data_Base_Time_parttion_realscale_group_" + str(groupnr) + ".pkl")

    with xr.open_dataset(real_action_dir + "interp-alt0-fileslide_0-regroup-" + str(groupnr) + ".nc") as ds:
        k_gr    = ds.k.data
        l_gr    = ds.l.data
        m_gr    = ds.m.data
        ntri_gr = ds.ntri.data

    N_real = np.zeros((2048, 145, len(l_gr), len(k_gr), len(m_gr)))
    for fileslide in range(60, 73):
        with xr.open_dataset(real_action_dir + "interp-alt0-fileslide_" + str(fileslide) + "-regroup-" + str(groupnr) + ".nc") as ds:
            tim = ds.time.data
            action = ds.data.data
        start_idx = 12 * (fileslide - 60)
        end_idx   = start_idx + len(tim)
        N_real[:, start_idx:end_idx, ...] = np.transpose(action, (1, 0, 2, 3, 4))

    print(f"Real data range for group {groupnr}: min={N_real.min():.4e}, max={N_real.max():.4e}, mean={N_real.mean():.4e}")

    MSE_score = np.zeros((2048, 145))
    Ratio_score = np.zeros((2048, 145))
    for orography in [True, False]:
        if orography:
            fname_pca = "PCA_spectrum_realscale_to_realscale_orography_" + str(groupnr) + ".nc"
            ntri_or_nonor = Data_Base["ntri_orography"]
        else:
            fname_pca = "PCA_spectrum_realscale_to_realscale_nonorography_" + str(groupnr) + ".nc"
            ntri_or_nonor = Data_Base["ntri_nonorography"]

        with xr.open_dataset(pca_reconstructed_action_dir + fname_pca) as ds:
            Pca_action = ds.PCA_spectrum.data

        index_oro_nonoro = np.where(np.isin(ntri_gr, ntri_or_nonor))[0]
        print("Reconstructed range:", Pca_action.min(), Pca_action.max(), Pca_action.mean())
        print("Real data range:    ", N_real[index_oro_nonoro].min(),
              N_real[index_oro_nonoro].max(),
              N_real[index_oro_nonoro].mean())

        MSE = np.sum((Pca_action - N_real[index_oro_nonoro, ...]) ** 2, axis=(2, 3, 4))
        NMSE = MSE / np.sum(N_real[index_oro_nonoro, ...] ** 2, axis=(2, 3, 4))

        ratio = np.sum(Pca_action, axis=(2, 3, 4)) / np.sum(N_real[index_oro_nonoro, ...], axis=(2, 3, 4))
        MSE_score[index_oro_nonoro, ...] = 1 / (1 + NMSE)
        Ratio_score[index_oro_nonoro, ...] = ratio

        print(f"ratio for group {groupnr} with {'orography' if orography else 'non-orography'}: {np.median(ratio):.4f}")
        print(f"MSE score for group {groupnr} with {'orography' if orography else 'non-orography'}: {np.median(MSE_score[index_oro_nonoro, ...]):.4f}")
    return Ratio_score, MSE_score, ntri_gr


if __name__ == "__main__":
    paths = load_paths()
    Data_base_Dir   = paths["pca_data_base_dir"]
    real_action_dir = paths["real_action_dir"]
    avg_dir         = paths["avg_dir"]
    EOF_dir         = paths["eof_dir"]
    PCA_recon_dir   = paths["pca_recon_dir"]
    metrics_dir     = paths["flux_energy_metrics_dir"]

    Ratio_score_matrix = np.zeros((20480))
    MSE_score_reconstruction = np.zeros((20480))

    for groupnr in range(10):
        for orography in [True, False]:
            print(f"Processing group {groupnr} with {'orography' if orography else 'non-orography'}...")
            pca_info = joblib.load(Data_base_Dir + "PCA_info_realscale.pkl")
            n_90 = pca_info[groupnr]["n_90_oro" if orography else "n_90_nonoro"]
            y_test, ntri_or_nonor = read_test_data(Data_base_Dir, orography, groupnr)
            print(f"PCA coefficients range for group {groupnr} with {'orography' if orography else 'non-orography'}: min={y_test.min():.4e}, max={y_test.max():.4e}, mean={y_test.mean():.4e}")
            print()
            print(f"  Number of EOF modes retained (n_90): {n_90}")
            print(f"  Shape of PCA coefficients (y_test): {y_test.shape}")

            reconstruction(
                EOF_dir=EOF_dir, avrage_dir=avg_dir, groupnr=groupnr,
                pca_coff=y_test, orography=orography, n_90=n_90,
                pca_spectrm_dir=PCA_recon_dir, ntri_or=ntri_or_nonor)

        ratio_score, mse_score, ntri_gr = reconstruction_error(
            real_action_dir=real_action_dir, pca_reconstructed_action_dir=PCA_recon_dir,
            data_base_path=Data_base_Dir, groupnr=groupnr)

        Ratio_score_matrix[ntri_gr] = np.mean(ratio_score, axis=1)
        MSE_score_reconstruction[ntri_gr] = np.mean(mse_score, axis=1)

    np.save(metrics_dir + "MSE_PCA_logscale_to_realscale_Reconstruction.npy", MSE_score_reconstruction)
    np.save(metrics_dir + "Ratio_PCA_logscale_to_realscale_Reconstruction.npy", Ratio_score_matrix)
