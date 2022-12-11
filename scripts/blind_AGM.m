%setup
setup;

%set fidelity metric to be amplitude Gaussian
mode = 'AGM';

%set seed
rng(1234);

%add Gaussian noise
n = size(dp,1);
snr_value = 40;
sigma = sqrt(10^(-snr_value/10)*sum(dp(:))/(100*256^2));
dp1 = sqrt(dp) + randn(n,n,100)*sigma;
dp1 = dp1.^2;



%set parameters
param.lambda = 10;
param.beta1 = 0.25;
param.beta2 = 0.25;
param.batch_size = 10;
param.step_size = 2*sqrt(param.batch_size);
param.z_update_mode = 'rPIE';
param.omega_update_mode = 'rPIE';
param.gamma = 0.1;
param.maxiter = 600;
param.alpha = 0.8;
param.omega = probe0;
param.gamma_omega = 0.025;
param.probe_step_size = 2*sqrt(param.batch_size)*5e-4;

rng('shuffle')
%stochastic ADMM with isotropic TV
[agm_iso, agm_iso_objective] = official_stochastic_iso_blind_ADMM(N_obj, dp1, ind_b, param, mode);
[~,ciso] = compute_snr_blind(agm_iso, obj_true);
iso_ssim_result = ssim(abs(ciso*agm_iso), abs(obj_true));
angle_iso_ssim_result = compute_angle_ssim(ciso*agm_iso, obj_true);

%stochastic ADMM with AITV (alpha = 0.8)
[agm_AITV, agm_AITV_objective] = official_stochastic_AITV_blind_ADMM(N_obj, dp1, ind_b, param, mode);
[~,cAITV] = compute_snr_blind(agm_AITV, obj_true);
aitv_ssim_result = ssim(abs(cAITV*agm_AITV), abs(obj_true));
angle_aitv_ssim_result = compute_angle_ssim(cAITV*agm_AITV, obj_true);

%plot figures
figure; subplot(2,3,1); imagesc(abs(obj_true)); axis off; axis image; colormap gray; title('Original Mag.')
subplot(2,3,2); imagesc(abs(agm_iso)); axis off; axis image; colormap gray; title('isoTV Mag.')
subplot(2,3,3); imagesc(abs(agm_AITV)); axis off; axis image; colormap gray; title('AITV Mag.')
subplot(2,3,4); imagesc(angle(obj_true)); axis off; axis image; colormap gray; title('Original Phase')
subplot(2,3,5); imagesc(angle(ciso*agm_iso), [0,0.4]); axis off; axis image; colormap gray; title('isoTV Phase')
subplot(2,3,6); imagesc(angle(cAITV*agm_AITV),[0,0.4]); axis off; axis image; colormap gray; title('AITV Phase')
