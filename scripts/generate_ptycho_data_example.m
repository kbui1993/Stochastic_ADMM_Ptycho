% load other images
boat = imread('boat.jpg');
house = imread('house.jpg');

% generate ptychography data from loaded images
mag_image = double(boat);
phase_image = double(house);
N_scans = 10;
rng(1234);
diffraction_size = 100; 
[obj, probe_true, dp, ind_b, dx] = generate_ptycho_data(mag_image, phase_image, N_scans, diffraction_size);


% get size of true image
N_obj = size(obj,1);

% get size of probe
Np = size(dp,1);

%set of constants
energy = 8.8;
lambda = 1.23984193e-9/energy;
dia_pert=1;

%generate initial probe
[probe0] =  generate_probe(Np, lambda, dx, Np*1e-6*dia_pert, 'velo');

%set parameters
param.lambda = 10;
param.beta1 = 0.25;
param.beta2 = 0.25;
param.batch_size = 20;
param.step_size = 2*sqrt(param.batch_size);
param.z_update_mode = 'rPIE';
param.omega_update_mode = 'rPIE';
param.gamma = 0.1;
param.maxiter = 600;
param.alpha = 0.8;
param.omega = probe0;
param.gamma_omega = 0.025;
param.probe_step_size = 2*sqrt(param.batch_size)*5e-4;

%stochastic ADMM with AITV (alpha = 0.8)
[agm_AITV, agm_AITV_objective] = official_stochastic_AITV_blind_ADMM(N_obj, dp, ind_b, param, 'AGM');
[~,cAITV] = compute_snr_blind(agm_AITV, obj);
aitv_ssim_result = ssim(abs(cAITV*agm_AITV), abs(obj));
angle_aitv_ssim_result = compute_angle_ssim(cAITV*agm_AITV, obj);

%plot figures
figure; subplot(2,2,1); imagesc(abs(obj)); axis off; axis image; colormap gray; title('Original Mag.')
subplot(2,2,2); imagesc(abs(agm_AITV)); axis off; axis image; colormap gray; title('AITV Mag.')
subplot(2,2,3); imagesc(angle(obj)); axis off; axis image; colormap gray; title('Original Phase')
subplot(2,2,4); imagesc(angle(cAITV*agm_AITV)); axis off; axis image; colormap gray; title('AITV Phase')

