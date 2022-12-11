%setup
setup;

%set amplitude to be intesnity Poisson metric
mode = 'IPM';

%set seed
rng(1234);

%add Poisson noise
dp1 = zeros(size(dp));
nu = 0.01;
obj_scale = nu*obj_true;
for i = 1:100
   obj_mask = obj_scale(ind_b(i,3):ind_b(i,4), ind_b(i,1):ind_b(i,2));
   Pobj = probe_true.*obj_mask;
   FPobj = fftshift(fft2(ifftshift(Pobj)));
   mag = abs(FPobj).^2;
   dp1(:,:,i) = poissrnd(mag);
end

%set parameters
param.lambda = 0.15;
param.beta1 = 0.25;
param.beta2 = 0.25;
param.z_update_mode = 'ePIE';
param.maxiter = 300;
param.alpha = 0.8;
param.init_z = nu*(ones(N_obj, N_obj)+1i*ones(N_obj, N_obj))/sqrt(2);
param.batch_size = 10;
param.step_size = 15*sqrt(param.batch_size);


rng('shuffle')
%stochastic ADMM with isotropic TV
[ipm_iso, ipm_iso_objective] = official_stochastic_iso_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);
[~,ciso] = compute_snr_blind(ipm_iso/nu, obj_true);
iso_ssim_result = ssim(abs(ciso*ipm_iso/nu), abs(obj_true));
angle_iso_ssim_result = compute_angle_ssim(ciso*ipm_iso, obj_true);

%stochastic ADMM with AITV (alpha = 0.8)
[ipm_AITV, ipm_AITV_objective] = official_stochastic_AITV_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);
[~,cAITV] = compute_snr_blind(ipm_AITV/nu, obj_true);
aitv_ssim_result = ssim(abs(cAITV*ipm_AITV/nu), abs(obj_true));
angle_aitv_ssim_result = compute_angle_ssim(cAITV*ipm_AITV, obj_true);

%plot figures
figure; subplot(2,3,1); imagesc(abs(obj_true)); axis off; axis image; colormap gray; title('Original Mag.')
subplot(2,3,2); imagesc(abs(ipm_iso)); axis off; axis image; colormap gray; title('isoTV Mag.')
subplot(2,3,3); imagesc(abs(ipm_AITV)); axis off; axis image; colormap gray; title('AITV Mag.')
subplot(2,3,4); imagesc(angle(obj_true)); axis off; axis image; colormap gray; title('Original Phase')
subplot(2,3,5); imagesc(angle(ciso*ipm_iso), [0,0.4]); axis off; axis image; colormap gray; title('isoTV Phase')
subplot(2,3,6); imagesc(angle(cAITV*ipm_AITV),[0,0.4]); axis off; axis image; colormap gray; title('AITV Phase')
