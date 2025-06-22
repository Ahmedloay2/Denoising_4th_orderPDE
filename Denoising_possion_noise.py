import os
os.environ['QT_QPA_FONTDIR'] = ''

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_laplace
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt
import time
from bm3d import bm3d
from concurrent.futures import ThreadPoolExecutor
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def add_poisson_noise(image, noise_level):
    base_scaling_factor = 100
    scaling_factors = {
        1: base_scaling_factor * 10,
        2: base_scaling_factor,
        3: base_scaling_factor / 2,
        4: base_scaling_factor / 10,
        5: base_scaling_factor / 20,
        6: base_scaling_factor / 100,
        7: base_scaling_factor / 1000,
        8: base_scaling_factor / 5000,
    }
    if np.max(image) > 1 or np.min(image) < 0:
        raise ValueError("Input image must be normalized to the range [0, 1].")
    if noise_level < 1 or noise_level > 8:
        raise ValueError("Noise level must be between 1 and 8.")
    factor = scaling_factors[noise_level]
    photon_image = image * factor
    noisy_image = np.random.poisson(photon_image) / factor
    return np.clip(noisy_image, 0, 1)

def compute_mad(image):
    mean = np.mean(image)
    return np.median(np.abs(image - mean))

def calculate_regularization_parameter(image):
    mu = np.mean(image)
    sigma = np.sqrt(np.var(image))
    return sigma / (mu + 1e-10)

def rfpde(image, noisy_image, lambda_val, num_iter, delta_t, k_mad):
    output = image.copy()
    for _ in range(num_iter):
        fidelity_term = (noisy_image - output) / (output + 1e-10) / lambda_val
        laplacian = gaussian_laplace(output, sigma=1)
        diffusion_coeff = 1 / (1 + (laplacian / k_mad) ** 2)
        modified_bilaplacian = gaussian_laplace(diffusion_coeff * laplacian, sigma=1)
        output += delta_t * (fidelity_term + modified_bilaplacian)
    return np.clip(output, 0, 1)

def optimize_with_aco(image, noisy_image, max_iter=10, num_ants=5):
    best_psnr = -np.inf
    best_params = None
    delta_t = 0.01
    for _ in range(max_iter):
        for _ in range(num_ants):
            lambda_val = calculate_regularization_parameter(noisy_image)
            num_iter = np.random.randint(10, 100)
            k_mad = compute_mad(noisy_image)
            denoised = rfpde(image, noisy_image, lambda_val, num_iter, delta_t, k_mad)
            score = psnr(image, denoised, data_range=1.0)
            if score > best_psnr:
                best_psnr = score
                best_params = (lambda_val, num_iter)
    return best_params

def process_rgb_image(original_rgb, noised, noise_level):
    start_time = time.perf_counter()
    def process_channel(args):
        channel, noised, noise_level = args
        noisy_channel = add_poisson_noise(channel, noise_level) if not noised or noise_level == 0 else channel
        lambda_val, num_iter = optimize_with_aco(channel, noisy_channel)
        k_mad = compute_mad(noisy_channel)
        delta_t = 0.01
        denoised_channel = rfpde(channel, noisy_channel, lambda_val, num_iter, delta_t, k_mad)
        return noisy_channel, denoised_channel
    args_list = [(original_rgb[:, :, c], noised, noise_level) for c in range(3)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(process_channel, args_list))
    noisy_rgb = np.stack([result[0] for result in results], axis=-1)
    denoised_rgb = np.stack([result[1] for result in results], axis=-1)
    end_time = time.perf_counter()
    print(f"process_rgb_image Execution Time: {end_time - start_time:.6f} seconds")
    return noisy_rgb, denoised_rgb

def bm3d_denoising(denoised_rgb):
    sigma_psd = 0.1
    return np.stack([bm3d(denoised_rgb[:, :, i], sigma_psd) for i in range(3)], axis=-1)

def mse(uref, uacorfpde, data_range=None):
    if len(uref) != len(uacorfpde):
        raise ValueError("Input arrays must have the same length.")
    uref = np.array(uref)
    uacorfpde = np.array(uacorfpde)
    if data_range is None:
        data_range = uref.max() - uref.min()
    return np.mean((uacorfpde - uref) ** 2)

def psnr(uref, uACORFPDE, data_range=None):
    if data_range is None:
        data_range = uref.max() - uref.min()
    mse_value = mse(uref, uACORFPDE)
    if mse_value == 0:
        return float('inf')
    return 10 * np.log10((data_range) ** 2 / mse_value)

def uqi(original, denoised, data_range=None):
    if len(original) != len(denoised):
        raise ValueError("Input arrays must have the same length.")
    original = np.array(original)
    denoised = np.array(denoised)
    if data_range is None:
        data_range = original.max() - original.min()
    mu_x = np.mean(original)
    mu_y = np.mean(denoised)
    sigma_x = np.std(original)
    sigma_y = np.std(denoised)
    sigma_xy = np.cov(original.flatten(), denoised.flatten())[0][1]
    numerator = 4 * sigma_xy * mu_x * mu_y
    denominator = (sigma_x ** 2 + sigma_y ** 2) * (mu_x ** 2 + mu_y ** 2)
    if denominator == 0:
        raise ValueError("Denominator is zero; check your input data for constant values.")
    return numerator / denominator

def nae(uref, u_acorfpde, data_range=None):
    if uref.shape != u_acorfpde.shape:
        raise ValueError("Input arrays must have the same shape.")
    uref = np.array(uref)
    u_acorfpde = np.array(u_acorfpde)
    if data_range is None:
        data_range = uref.max() - uref.min()
    numerator = np.sum(np.abs(uref - u_acorfpde))
    denominator = np.sum(np.abs(uref))
    if denominator == 0:
        raise ValueError("The denominator (sum of |uref|) is zero.")
    return (numerator / denominator)

def cc(uref, uacorfpde, data_range=None):
    if len(uref) != len(uacorfpde):
        raise ValueError("Input arrays must have the same length.")
    uref = np.array(uref)
    uacorfpde = np.array(uacorfpde)
    if data_range is None:
        data_range = uref.max() - uref.min()
    mu_ref = np.mean(uref)
    mu_uacorfpde = np.mean(uacorfpde)
    numerator = np.sum((uref - mu_ref) * (uacorfpde - mu_uacorfpde))
    denominator = np.sqrt(np.sum((uref - mu_ref) ** 2) * np.sum((uacorfpde - mu_uacorfpde) ** 2))
    if denominator == 0:
        raise ValueError("Denominator is zero; check your input data for constant values.")
    return numerator / denominator

def visualize_rgb_results(original, noisy, denoised, final_denoised, psnr_original, ssim_original, psnr_noised, ssim_noised, psnr_pde, ssim_pde, psnr_final, ssim_final):
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.title(f"Original RGB Image\nPSNR: {psnr_original:.2f}, SSIM: {ssim_original:.2f}")
    plt.imshow(original)
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.title(f"Noisy RGB Image\nPSNR: {psnr_noised:.2f}, SSIM: {ssim_noised:.2f}")
    plt.imshow(noisy)
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.title(f"PDE Denoised RGB Image\nPSNR: {psnr_pde:.2f}, SSIM: {ssim_pde:.2f}")
    plt.imshow(denoised)
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.title(f"BM3D Final Denoised RGB Image\nPSNR: {psnr_final:.2f}, SSIM: {ssim_final:.2f}")
    plt.imshow(final_denoised)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_metrics_without_psnr(ssim_noised, mse_noised, uqi_noised, nae_noised, cc_noised,
                              ssim_pde, mse_pde, uqi_pde, nae_pde, cc_pde,
                              ssim_final, mse_final, uqi_final, nae_final, cc_final):
    stages = ['Noised', 'PDE Denoised', 'Final']
    metrics = {
        'MSE': [mse_noised, mse_pde, mse_final],
        'CC': [cc_noised, cc_pde, cc_final],
        'NAE': [nae_noised, nae_pde, nae_final],
        'UQI': [uqi_noised, uqi_pde, uqi_final],
        'SSIM': [ssim_noised, ssim_pde, ssim_final]
    }
    markers = ['s', 'o', '^', 'v', 'D']
    colors = ['black', 'red', 'blue', 'green', 'purple']
    plt.figure(figsize=(10, 6))
    for i, (metric, values) in enumerate(metrics.items()):
        plt.plot(stages, values, marker=markers[i], color=colors[i], label=metric, linestyle='-')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Stages', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.title('Comparison of Metrics (Excluding PSNR) Across Stages', fontsize=14)
    plt.legend(title="Metrics", fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_psnr_only(psnr_noised, psnr_pde, psnr_final):
    stages = ['Noised', 'PDE Denoised', 'Final']
    psnr_values = [psnr_noised, psnr_pde, psnr_final]
    plt.figure(figsize=(8, 5))
    plt.plot(stages, psnr_values, marker='o', color='blue', label='PSNR', linestyle='-')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Stages', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('PSNR Comparison Across Stages', fontsize=14)
    plt.legend(title="Metric", fontsize=10)
    plt.tight_layout()
    plt.show()

def visualize_3d_image(image, title, global_min, global_max):
    norm = plt.Normalize(global_min, global_max)
    cmap = plt.cm.Blues  # Change to blue-to-red color map

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    x, y = np.meshgrid(x, y)

    z = np.mean(image, axis=2)
    z = (z - global_min) / (global_max - global_min)  # Normalize to [0, 1]
    z = np.clip(z, 0, 1)  # Ensure all z values are within [0, 1]
    surf = ax.plot_surface(x, y, z, facecolors=cmap(z), edgecolor='none')

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Width", fontsize=10)
    ax.set_ylabel("Height", fontsize=10)
    ax.set_zlabel("Normalized Intensity", fontsize=10)
    ax.set_zlim3d(0, 1)  # Ensure z-axis is scaled to [0, 1]
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.tick_params(axis='y', labelsize=8, rotation=45)
    ax.tick_params(axis='z', labelsize=8)

    # Add a color bar
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(np.linspace(global_min, global_max, 100))  # Ensure full color range
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Normalized Intensity")

    plt.tight_layout()
    plt.show()

def visualize_3d_comparison(original, noisy, pde_denoised, bm3d_denoised, noise_level):
    images = [
        ("Original", original),
        ("Noisy", noisy),
        ("PDE Denoised", pde_denoised),
        ("BM3D Denoised", bm3d_denoised)
    ]

    global_min = float("inf")
    global_max = float("-inf")
    for _, img in images:
        z = np.mean(img, axis=2)
        global_min = min(global_min, z.min())
        global_max = max(global_max, z.max())

    for title, img in images:
        visualize_3d_image(img, title, global_min, global_max)

def generate_results_table(results, noise_level):
    rows = []
    for noise_level, data in results.items():
        rows.append([
            noise_level,
            data["psnr_noised"], data["ssim_noised"], data["mse_noised"], data["uqi_noised"], data["nae_noised"], data["cc_noised"],
            data["psnr_pde"], data["ssim_pde"], data["mse_pde"], data["uqi_pde"], data["nae_pde"], data["cc_pde"],
            data["psnr_final"], data["ssim_final"], data["mse_final"], data["uqi_final"], data["nae_final"], data["cc_final"]
        ])
    columns = [
        "Noise Level",
        "PSNR (Noised)", "SSIM (Noised)", "MSE (Noised)", "UQI (Noised)", "NAE (Noised)", "CC (Noised)",
        "PSNR (PDE)", "SSIM (PDE)", "MSE (PDE)", "UQI (PDE)", "NAE (PDE)", "CC (PDE)",
        "PSNR (Final)", "SSIM (Final)", "MSE (Final)", "UQI (Final)", "NAE (Final)", "CC (Final)"
    ]
    return pd.DataFrame(rows, columns=columns)

def main():
    start_time = time.perf_counter()
    image_path = 'No-noise.jpg'
    original_rgb = cv2.imread(image_path)
    if original_rgb is None:
        print("Error: Image not found. Please check the path.")
        return
    original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB) / 255.0
    print("Original RGB image loaded and normalized.")
    noise_levels = [4, 6]
    results = {}
    for noise_level in noise_levels:
        print(f"Processing for Noise Level {noise_level}...")
        noisy_rgb, pde_denoised_rgb = process_rgb_image(original_rgb, noised=False, noise_level=noise_level)
        final_denoised_rgb = bm3d_denoising(pde_denoised_rgb)
        print(f"Final Denoised RGB image generated for Noise Level {noise_level} using BM3D.")
        psnr_noised = np.mean([psnr(original_rgb[:, :, c], noisy_rgb[:, :, c], data_range=1.0) for c in range(3)])
        ssim_noised = np.mean([ssim(original_rgb[:, :, c], noisy_rgb[:, :, c], data_range=1.0) for c in range(3)])
        mse_noised = np.mean([mse(original_rgb[:, :, c], noisy_rgb[:, :, c], data_range=1.0) for c in range(3)])
        uqi_noised = np.mean([uqi(original_rgb[:, :, c], noisy_rgb[:, :, c], data_range=1.0) for c in range(3)])
        nae_noised = np.mean([nae(original_rgb[:, :, c], noisy_rgb[:, :, c], data_range=1.0) for c in range(3)])
        cc_noised = np.mean([cc(original_rgb[:, :, c], noisy_rgb[:, :, c], data_range=1.0) for c in range(3)])
        psnr_pde = np.mean([psnr(original_rgb[:, :, c], pde_denoised_rgb[:, :, c], data_range=1.0) for c in range(3)])
        ssim_pde = np.mean([ssim(original_rgb[:, :, c], pde_denoised_rgb[:, :, c], data_range=1.0) for c in range(3)])
        mse_pde = np.mean([mse(original_rgb[:, :, c], pde_denoised_rgb[:, :, c], data_range=1.0) for c in range(3)])
        uqi_pde = np.mean([uqi(original_rgb[:, :, c], pde_denoised_rgb[:, :, c], data_range=1.0) for c in range(3)])
        nae_pde = np.mean([nae(original_rgb[:, :, c], pde_denoised_rgb[:, :, c], data_range=1.0) for c in range(3)])
        cc_pde = np.mean([cc(original_rgb[:, :, c], pde_denoised_rgb[:, :, c], data_range=1.0) for c in range(3)])
        psnr_final = np.mean([psnr(original_rgb[:, :, c], final_denoised_rgb[:, :, c], data_range=1.0) for c in range(3)])
        ssim_final = np.mean([ssim(original_rgb[:, :, c], final_denoised_rgb[:, :, c], data_range=1.0) for c in range(3)])
        mse_final = np.mean([mse(original_rgb[:, :, c], final_denoised_rgb[:, :, c], data_range=1.0) for c in range(3)])
        uqi_final = np.mean([uqi(original_rgb[:, :, c], final_denoised_rgb[:, :, c], data_range=1.0) for c in range(3)])
        nae_final = np.mean([nae(original_rgb[:, :, c], final_denoised_rgb[:, :, c], data_range=1.0) for c in range(3)])
        cc_final = np.mean([cc(original_rgb[:, :, c], final_denoised_rgb[:, :, c], data_range=1.0) for c in range(3)])
        results[noise_level] = {
            "original": original_rgb,
            "noisy": noisy_rgb,
            "pde_denoised": pde_denoised_rgb,
            "final_denoised": final_denoised_rgb,
            "psnr_noised": psnr_noised,
            "ssim_noised": ssim_noised,
            "mse_noised": mse_noised,
            "uqi_noised": uqi_noised,
            "nae_noised": nae_noised,
            "cc_noised": cc_noised,
            "psnr_pde": psnr_pde,
            "ssim_pde": ssim_pde,
            "mse_pde": mse_pde,
            "uqi_pde": uqi_pde,
            "nae_pde": nae_pde,
            "cc_pde": cc_pde,
            "psnr_final": psnr_final,
            "ssim_final": ssim_final,
            "mse_final": mse_final,
            "uqi_final": uqi_final,
            "nae_final": nae_final,
            "cc_final": cc_final
        }
        visualize_3d_comparison(original_rgb, noisy_rgb, pde_denoised_rgb, final_denoised_rgb, noise_level)
    for noise_level, result in results.items():
        print(f"Visualizing results for Noise Level {noise_level}...")
        visualize_rgb_results(
            original_rgb,
            result["noisy"],
            result["pde_denoised"],
            result["final_denoised"],
            psnr_original=np.mean([psnr(original_rgb[:, :, c], original_rgb[:, :, c], data_range=1.0) for c in range(3)]),
            ssim_original=np.mean([ssim(original_rgb[:, :, c], original_rgb[:, :, c], data_range=1.0) for c in range(3)]),
            psnr_noised=result["psnr_noised"],
            ssim_noised=result["ssim_noised"],
            psnr_pde=result["psnr_pde"],
            ssim_pde=result["ssim_pde"],
            psnr_final=result["psnr_final"],
            ssim_final=result["ssim_final"],
        )
        print("Graph for PSNR for Noise Level {noise_level}...")
        plot_psnr_only(
            psnr_noised=result["psnr_noised"],
            psnr_pde=result["psnr_pde"],
            psnr_final=result["psnr_final"]
        )
        print(f"Generating graph for Noise Level {noise_level}...")
        plot_metrics_without_psnr(
            ssim_noised=result["ssim_noised"],
            mse_noised=result["mse_noised"],
            uqi_noised=result["uqi_noised"],
            nae_noised=result["nae_noised"],
            cc_noised=result["cc_noised"],
            ssim_pde=result["ssim_pde"],
            mse_pde=result["mse_pde"],
            uqi_pde=result["uqi_pde"],
            nae_pde=result["nae_pde"],
            cc_pde=result["cc_pde"],
            ssim_final=result["ssim_final"],
            mse_final=result["mse_final"],
            uqi_final=result["uqi_final"],
            nae_final=result["nae_final"],
            cc_final=result["cc_final"],
            )
    end_time = time.perf_counter()
    print(f"Script Execution Time: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    main()
