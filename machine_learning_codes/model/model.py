import os

import numpy as np
import xarray as xr
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, Activation,
    GRU, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2

from src.config import load_paths


def read_scale_data(model_save_dir, coarse_action_dir_jan, land_ocean_tag="ocean", physics=False):
    print("Loading databases...")

    paths = load_paths()
    data_base_path_jan = paths["data_base_dir_jan"]
    data_base_path_aug = paths["data_base_dir_aug"]

    Data_Base_jan = joblib.load(
        data_base_path_jan + "corsed_action_" + land_ocean_tag + ".pkl"
    )
    Data_Base_aug = joblib.load(
        data_base_path_aug + "corsed_action_" + land_ocean_tag + ".pkl"
    )

    X_train = np.concatenate([Data_Base_jan["X_train"], Data_Base_aug["X_train"]], axis=1)
    X_val   = np.concatenate([Data_Base_jan["X_val"],   Data_Base_aug["X_val"]],   axis=1)
    X_test  = np.concatenate([Data_Base_jan["X_test"],  Data_Base_aug["X_test"]],  axis=1)
    
    cgz_train = np.concatenate([Data_Base_jan["cgz_train"], Data_Base_aug["cgz_train"]], axis=1)
    cgz_val = np.concatenate([Data_Base_jan["cgz_val"], Data_Base_aug["cgz_val"]], axis=1)
    cgz_test = np.concatenate([Data_Base_jan["cgz_test"], Data_Base_aug["cgz_test"]], axis=1)
    omega_hat_train = np.concatenate([Data_Base_jan["omega_hat_train"], Data_Base_aug["omega_hat_train"]], axis=1)
    omega_hat_val = np.concatenate([Data_Base_jan["omega_hat_val"], Data_Base_aug["omega_hat_val"]], axis=1)
    omega_hat_test = np.concatenate([Data_Base_jan["omega_hat_test"], Data_Base_aug["omega_hat_test"]], axis=1)

    with xr.open_dataset(coarse_action_dir_jan + "coarse_gw_spectrum_group_1_fileslide_0.nc") as ds:
        k_lunch = ds.k.data
        l_lunch = ds.l.data
        m_lunch = -ds.m.data


    y_train = np.concatenate([Data_Base_jan["y_train"], Data_Base_aug["y_train"]], axis=1)
    y_val   = np.concatenate([Data_Base_jan["y_val"],   Data_Base_aug["y_val"]],   axis=1)
    y_test  = np.concatenate([Data_Base_jan["y_test"],  Data_Base_aug["y_test"]],  axis=1)

    y_train = np.nan_to_num(y_train, nan=0.0)
    y_val   = np.nan_to_num(y_val, nan=0.0)
    y_test  = np.nan_to_num(y_test, nan=0.0)

    ntri = Data_Base_jan["ntri"]
    n_features = X_train.shape[-1]
    n_targets = y_train.shape[2] * y_train.shape[3] * y_train.shape[4]  # 6 * 6 * 4 = 144

    print(f"Combined shapes → X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Combined shapes → X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f"Combined shapes → X_val:   {X_val.shape}, y_val:   {y_val.shape}")

    X_train = X_train.reshape(-1, n_features)
    X_val   = X_val.reshape(-1, n_features)
    X_test  = X_test.reshape(-1, n_features)

    y_train = y_train.reshape(-1, n_targets)
    y_val   = y_val.reshape(-1, n_targets)
    y_test  = y_test.reshape(-1, n_targets)
    
    
    y_train = np.log1p(y_train)
    y_val   = np.log1p(y_val)
    y_test  = np.log1p(y_test)

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val   = x_scaler.transform(X_val)
    X_test  = x_scaler.transform(X_test)

    joblib.dump(x_scaler, f"{model_save_dir}scaler/scaler_X_{land_ocean_tag}.pkl")

    return (X_train, X_val, y_train, y_val, X_test, y_test, x_scaler, ntri, cgz_train.reshape(-1, n_targets), cgz_val.reshape(-1, n_targets),
             cgz_test.reshape(-1, n_targets), omega_hat_train.reshape(-1, n_targets), omega_hat_val.reshape(-1, n_targets), omega_hat_test.reshape(-1, n_targets),
            k_lunch.reshape(-1, 1, 1), l_lunch.reshape(1, -1, 1), m_lunch.reshape(1, 1, -1))


def read_scale_data_lag(data_base_path, model_save_dir, land_ocean_tag="ocean"):
    print("Loading database...")

    Data_Base = joblib.load(
        data_base_path + "Data_Base_Time_parttion_coarsedN_lag_" + land_ocean_tag + ".pkl"
    )

    X_train = Data_Base["X_train"]
    X_val   = Data_Base["X_val"]
    X_test  = Data_Base["X_test"]

    y_train = Data_Base["y_train"]
    y_val   = Data_Base["y_val"]
    y_test  = Data_Base["y_test"]

    ntri = Data_Base["ntri"]

    print(f"Original X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Original X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"Original X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    n_lag      = X_train.shape[2]
    n_features = X_train.shape[3]
    n_targets  = y_train.shape[-1]

    def flatten_lag_samples(X, y):
        n_tri, n_time, n_lag_local, n_feat_local = X.shape
        assert n_lag_local == n_lag
        assert n_feat_local == n_features
        X_flat = X.reshape(n_tri * n_time, n_lag_local * n_feat_local)
        y_flat = y.reshape(n_tri * n_time, n_targets)
        return X_flat, y_flat

    X_train, y_train = flatten_lag_samples(X_train, y_train)
    X_val,   y_val   = flatten_lag_samples(X_val,   y_val)
    X_test,  y_test  = flatten_lag_samples(X_test,  y_test)

    y_train = np.log1p(y_train)
    y_val   = np.log1p(y_val)
    y_test  = np.log1p(y_test)

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val   = x_scaler.transform(X_val)
    X_test  = x_scaler.transform(X_test)

    joblib.dump(x_scaler, f"{model_save_dir}scaler/scaler_X_lag_{land_ocean_tag}.pkl")

    print(f"Flattened X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Flattened X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"Flattened X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    return X_train, X_val, y_train, y_val, X_test, y_test, x_scaler, ntri


def read_scale_data_lag_gru(data_base_path, model_save_dir, land_ocean_tag="ocean"):
    print("Loading database...")

    Data_Base = joblib.load(
        data_base_path + "Data_Base_Time_parttion_coarsedN_lag_" + land_ocean_tag + ".pkl"
    )

    X_train = Data_Base["X_train"]
    X_val   = Data_Base["X_val"]
    X_test  = Data_Base["X_test"]

    y_train = Data_Base["y_train"]
    y_val   = Data_Base["y_val"]
    y_test  = Data_Base["y_test"]

    ntri = Data_Base["ntri"]

    print(f"Original X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Original X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"Original X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    n_lag      = X_train.shape[2]  # 6 lags
    n_features = X_train.shape[3]
    n_targets  = y_train.shape[-1]

    def reshape_for_gru(X, y):
        n_tri, n_time, n_lag_local, n_feat_local = X.shape
        X_seq = X.reshape(n_tri * n_time, n_lag_local, n_feat_local)
        y_flat = y.reshape(n_tri * n_time, n_targets)
        return X_seq, y_flat

    X_train, y_train = reshape_for_gru(X_train, y_train)
    X_val,   y_val   = reshape_for_gru(X_val,   y_val)
    X_test,  y_test  = reshape_for_gru(X_test,  y_test)

    y_train = np.log1p(y_train)
    y_val   = np.log1p(y_val)
    y_test  = np.log1p(y_test)

    def fit_transform_3d(scaler, X, fit=True):
        n_samples, n_lag_local, n_feat_local = X.shape
        X_2d = X.reshape(-1, n_feat_local)
        X_2d = scaler.fit_transform(X_2d) if fit else scaler.transform(X_2d)
        return X_2d.reshape(n_samples, n_lag_local, n_feat_local)

    x_scaler = StandardScaler()
    X_train = fit_transform_3d(x_scaler, X_train, fit=True)
    X_val   = fit_transform_3d(x_scaler, X_val,   fit=False)
    X_test  = fit_transform_3d(x_scaler, X_test,  fit=False)

    joblib.dump(x_scaler, f"{model_save_dir}scaler/scaler_X_lag_{land_ocean_tag}.pkl")

    print(f"GRU-ready X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"GRU-ready X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"GRU-ready X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f"input_shape for GRU → ({n_lag}, {n_features})")

    return X_train, X_val, y_train, y_val, X_test, y_test, x_scaler, ntri


def make_tf_dataset(X, y, batch_size, shuffle=True, repeat=True):
    ds = tf.data.Dataset.from_tensor_slices((
        tf.cast(X, tf.float32),
        tf.cast(y, tf.float32)
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    if repeat:
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_tf_dataset_physics(X, y, cgz, omega_hat, batch_size, shuffle=True, repeat=True):
    y_combined = np.concatenate([y, cgz, omega_hat], axis=1)
    ds = tf.data.Dataset.from_tensor_slices((
        tf.cast(X, tf.float32),
        tf.cast(y_combined, tf.float32)
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    if repeat:
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_tf_dataset_weighted(X, y, sample_weight, batch_size, shuffle=True, repeat=True):
    ds = tf.data.Dataset.from_tensor_slices((
        tf.cast(X, tf.float32),
        tf.cast(y, tf.float32),
        tf.cast(sample_weight, tf.float32)
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    if repeat:
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def physics_flux_loss(k, l, n_targets=144, coff_wave_action=0.5, coff_flux_x=0, coff_flux_y=0.0, energy_coff=0.5):
    k = tf.constant(k, dtype=tf.float32)
    l = tf.constant(l, dtype=tf.float32)

    def loss(y_combined_true, y_pred):
        y_true    = tf.reshape(y_combined_true[:, :n_targets], (-1, 6, 6, 4))
        cgz_batch = tf.reshape(y_combined_true[:, n_targets:2*n_targets], (-1, 6, 6, 4))
        omega_hat_batch = tf.reshape(y_combined_true[:, 2*n_targets:], (-1, 6, 6, 4))

        N_true = tf.math.expm1(y_true)
        y_pred = tf.reshape(y_pred, (-1, 6, 6, 4))
        N_pred = tf.math.expm1(y_pred)

        uw_true = tf.reduce_sum(N_true * k[None, :, None, None] * cgz_batch, axis=(1, 2, 3))
        uw_pred = tf.reduce_sum(N_pred * k[None, :, None, None] * cgz_batch, axis=(1, 2, 3))
        vw_true = tf.reduce_sum(N_true * l[None, None, :, None] * cgz_batch, axis=(1, 2, 3))
        vw_pred = tf.reduce_sum(N_pred * l[None, None, :, None] * cgz_batch, axis=(1, 2, 3))
        E_true  = tf.reduce_sum(N_true*omega_hat_batch, axis=(1, 2, 3))
        E_pred  = tf.reduce_sum(N_pred*omega_hat_batch, axis=(1, 2, 3))

        uw_err = tf.reduce_mean(tf.square(uw_true - uw_pred)) / \
                   (tf.reduce_mean(tf.square(uw_true)) + 1e-9)
        vw_err = tf.reduce_mean(tf.square(vw_true - vw_pred)) / \
                           (tf.reduce_mean(tf.square(vw_true)) + 1e-9)
        E_err = tf.reduce_mean(tf.square(E_true - E_pred)) / \
                           (tf.square(E_true) + 1e-9)
        
        mse_raw  = tf.reduce_sum((y_true - y_pred)**2, axis=(1, 2, 3))
        mse_norm = tf.reduce_mean(mse_raw / (tf.reduce_sum(y_true**2, axis=(1, 2, 3)) + 1e-9))

        error =       coff_wave_action * mse_norm + coff_flux_x * uw_err + coff_flux_y * vw_err + energy_coff*E_err  
        return error
    return loss             


def model_architecture(input_shape, n_latent, training=False):
    inputs = Input(shape=input_shape)

    x = Dense(512)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    x = Dense(512)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    x = Dense(512, kernel_regularizer=l2(1e-4))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = Dense(512, kernel_regularizer=l2(1e-4))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = Dense(256)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    x = Dense(128)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    outputs = Dense(n_latent)(x)

    return Model(inputs, outputs)


def weighted_mae_loss(p90, tail_weight=2.0):
    def loss(y_true, y_pred):
        residuals = tf.abs(y_pred - y_true)
        weights = tf.where(
            y_true > p90,
            tf.ones_like(y_true) * tail_weight,
            tf.ones_like(y_true)
        )
        return tf.reduce_mean(weights * residuals)
    return loss

def MSE_loss_N():
    def metric(y_combined_true, y_pred):
        y_true = y_combined_true[:, :144]
        return tf.reduce_mean(tf.square(y_true - y_pred))
    metric.__name__ = "MSE"
    return metric


def model_architecture_DNN(input_shape, n_latent):
    inputs = Input(shape=input_shape)

    x = Dense(512)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    x = Dense(256, kernel_regularizer=l2(1e-4))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)

    x = Dense(128, kernel_regularizer=l2(1e-4))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.05)(x)

    x = Dense(64, kernel_regularizer=l2(1e-4))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    outputs = Dense(n_latent)(x)

    return Model(inputs, outputs)


def model_architecture_gru(input_shape, n_latent):
    """
    input_shape: (6, n_features)
    n_latent:    output size
    """
    inputs = Input(shape=input_shape)

    x = GRU(256, return_sequences=True, kernel_regularizer=l2(1e-4))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)

    x = GRU(128, return_sequences=True, kernel_regularizer=l2(1e-4))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)

    x = GRU(64, return_sequences=False, kernel_regularizer=l2(1e-4))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.05)(x)

    x = Dense(64, kernel_regularizer=l2(1e-4))(x)
    x = LayerNormalization()(x)
    x = Activation("relu")(x)

    outputs = Dense(n_latent)(x)

    return Model(inputs, outputs)


def model_architecture_gru_attn(input_shape, n_latent):
    inputs = Input(shape=input_shape)  # (6, 90)

    x = GRU(256, return_sequences=True, kernel_regularizer=l2(1e-4))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)

    x = GRU(128, return_sequences=True, kernel_regularizer=l2(1e-4))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)

    attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = LayerNormalization()(x + attn)

    x = GlobalAveragePooling1D()(x)

    x = Dense(128, kernel_regularizer=l2(1e-4))(x)
    x = LayerNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.1)(x)

    outputs = Dense(n_latent)(x)
    return Model(inputs, outputs)


def physics_mse_metric(n_targets=144):
    def mse_metric(y_combined_true, y_pred):
        y_true = y_combined_true[:, :n_targets]
        return tf.reduce_mean(tf.square(y_true - y_pred))
    mse_metric.__name__ = "mse_log"
    return mse_metric


def model_architecture_spectral_cnn(input_shape, n_k=6, n_l=6, n_m=4):
    n_latent = n_k * n_l * n_m

    inputs = Input(shape=input_shape)

    x = layers.Dense(256, kernel_regularizer=l2(10**-4))(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Dense(128, kernel_regularizer=l2(10**-4))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Dense(n_k * n_l * n_m * 16)(x)
    x = layers.Reshape((n_k, n_l, n_m, 16))(x)

    x = layers.Conv3D(16, kernel_size=(3, 3, 3), padding="same",
                       activation="relu", kernel_regularizer=l2(1e-3))(x)

    x = layers.Conv3D(4, kernel_size=(3, 3, 3), padding="same",
                       activation="relu", kernel_regularizer=l2(1e-3))(x)

    x = layers.Conv3D(1, kernel_size=(1, 1, 1), padding="same", activation = "softplus")(x)

    outputs = layers.Reshape((n_latent,))(x)

    return Model(inputs, outputs)


def physics_flux_metric(k3D_flat, n_targets=144, scale_to_mpa=1000.0):
    k3D_flat_tf = tf.constant(k3D_flat, dtype=tf.float32)

    def flux_rre(y_combined_true, y_pred):
        y_true    = y_combined_true[:, :n_targets]
        cgz_batch = y_combined_true[:, n_targets:]

        N_true = tf.math.expm1(y_true)
        N_pred = tf.math.expm1(y_pred)

        uw_true = tf.reduce_sum(N_true * k3D_flat_tf[None, :] * cgz_batch, axis=1) * scale_to_mpa
        uw_pred = tf.reduce_sum(N_pred * k3D_flat_tf[None, :] * cgz_batch, axis=1) * scale_to_mpa

        return tf.reduce_mean(tf.square(uw_true - uw_pred)) / (tf.reduce_mean(tf.square(uw_true)) + 1e-9)

    flux_rre.__name__ = "flux_rre_mpa"
    return flux_rre


def fit_save_model(X_train, X_val, y_train, y_val, k, l, save_dir_model, land_ocean_tag,
                    batch_size=125, epochs=50, arc="CNN", physics=False):

    if arc == "GRU":
        input_shape = (X_train.shape[1], X_train.shape[2])
    elif (arc == "CNN") or ("DNN"):
        input_shape = (X_train.shape[1],)

    n_latent    = y_train.shape[1]
    n_samples   = X_train.shape[0]
    y_var_const = float(np.mean(np.square(y_train)))
    print(f"y_var_const = {y_var_const}")

    print(f"Samples: {n_samples}, Input: {input_shape}, Latent: {n_latent}")
    if arc == "GRU":
        model = model_architecture_gru_attn(input_shape=input_shape, n_latent=n_latent)
    elif arc == "DNN":
        model = model_architecture_DNN(input_shape=input_shape, n_latent=n_latent)
    elif arc == "CNN":
        model = model_architecture_spectral_cnn(input_shape=input_shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=physics_flux_loss(k=k, l=l, n_targets=144),
        metrics=[MSE_loss_N()]
    )

    if arc == "GRU":
        best_model_path = f"{save_dir_model}/models/dnn_coarsed_lag_GRU_{land_ocean_tag}.keras"
    elif arc == "DNN":
        best_model_path = f"{save_dir_model}/models/dnn_coarsed_{land_ocean_tag}.keras"
    elif arc == "CNN":
        best_model_path = f"{save_dir_model}/models/cnn_pinn_coarsed_{land_ocean_tag}.keras"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path, monitor='val_MSE', save_best_only=True, mode='min', verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_MSE', patience=12, restore_best_weights=True, mode="min"
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_MSE', factor=0.5, patience=5, min_lr=1e-7, verbose=1, mode="min"
    )

    steps_per_epoch  = n_samples // batch_size
    validation_steps = X_val.shape[0] // batch_size
    if not physics:
        train_ds = make_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True, repeat=True)
        val_ds   = make_tf_dataset(X_val,   y_val,   batch_size=batch_size, shuffle=False, repeat=True)

    if physics:
        train_ds = make_tf_dataset_physics(X_train, y_train, cgz_train, omega_hat_train, batch_size=batch_size, shuffle=True, repeat=True)
        val_ds   = make_tf_dataset_physics(X_val, y_val, cgz_val, omega_hat_val, batch_size=batch_size, shuffle=False, repeat=True)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stop, lr_scheduler],
        verbose=1
    )

    return model, history


def build_flat_wavenumber_grids(k, l, m):
    # match the (l, k, m) broadcast order used in compute_intrinsic_feq_cgz / compute_cgz_batch
    k3D = np.broadcast_to(k[None, :, None], (len(l), len(k), len(m)))  # (6,6,4)
    l3D = np.broadcast_to(l[:, None, None], (len(l), len(k), len(m)))  # (6,6,4)

    k_flat = k3D.reshape(-1)
    l_flat = l3D.reshape(-1)
    return k_flat, l_flat


def continue_fit_save_model(X_train, X_val, y_train, y_val, save_dir_model, land_ocean_tag,
                             batch_size=125, completed_epochs=16, total_epochs=50, arc="CNN"):
    """
    Reload the best saved model and continue training.
    completed_epochs: starting epoch, e.g. 16.
    total_epochs: total desired epoch number, e.g. 50 means train from epoch 17 to 50.
    """
    if arc == "GRU":
        model_path = f"{save_dir_model}/models/dnn_coarsed_lag_GRU_{land_ocean_tag}.keras"
    elif arc == "DNN":
        model_path = f"{save_dir_model}/models/dnn_coarsed_{land_ocean_tag}.keras"
    elif arc == "CNN":
        model_path = f"{save_dir_model}/models/cnn_coarsed_{land_ocean_tag}.keras"
    else:
        raise ValueError(f"Unknown architecture: {arc}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    print(f"Loaded model from: {model_path}")
    print(f"Restored learning rate: "
          f"{float(tf.keras.backend.get_value(model.optimizer.learning_rate)):.2e}")
    print(f"Continuing from epoch {completed_epochs + 1} to {total_epochs}")

    train_ds = make_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True, repeat=True)
    val_ds   = make_tf_dataset(X_val,   y_val,   batch_size=batch_size, shuffle=False, repeat=True)

    steps_per_epoch = X_train.shape[0] // batch_size
    validation_steps = X_val.shape[0] // batch_size

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path, monitor="val_MSE", save_best_only=True, mode="min", verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_MSE", patience=12, restore_best_weights=True, mode="min", verbose=1
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_MSE", factor=0.5, patience=5, min_lr=1e-7, mode="min", verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=completed_epochs,
        epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stop, lr_scheduler],
        verbose=1
    )

    return model, history


if __name__ == "__main__":
    paths = load_paths()
    save_dir = paths["save_dir"]

    for tag in ["ocean"]:
        (X_train, X_val, y_train, y_val, X_test, y_test, X_scaler, ntri, cgz_train, cgz_val, _, 
         omega_hat_train, omega_hat_val, _, k, l, m) = read_scale_data(
            model_save_dir=save_dir,
            coarse_action_dir_jan=paths["coarse_action_dir_jan"],
            land_ocean_tag=tag
        )
        print(f"var of the train data {np.var(y_train)}")
        print(f"var of val {np.var(y_val)}")

        model, history = fit_save_model(
            X_train=X_train, X_val=X_val,
            y_train=y_train, y_val=y_val,
            k=k, l=l,
            save_dir_model=save_dir,
            land_ocean_tag=tag, arc="CNN", physics=True
        )