%setup
setup;
%setup_chip;

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
param.batch_size = 5;
param.step_size = 15*sqrt(param.batch_size);
param.z_update_mode = 'ePIE';
%param.gamma = 0.75;
param.maxiter = 300;
param.alpha = 0.8;
param.init_z = nu*(ones(N_obj, N_obj)+1i*ones(N_obj, N_obj))/sqrt(2);

rng('shuffle')

DR_z = nonblind_ptycho_DR(N_obj, sqrt(dp1), ind_b, probe_true, 15);


rng('shuffle')

[ipm_z5_iso, ipm_z5_iso_objective] = official_stochastic_iso_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);
[ipm_z5_AITV, ipm_z5_AITV_objective] = official_stochastic_AITV_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);

param.batch_size = 10;
param.step_size = 15*sqrt(param.batch_size);
[ipm_z10_iso, ipm_z10_iso_objective] = official_stochastic_iso_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);
[ipm_z10_AITV, ipm_z10_AITV_objective] = official_stochastic_AITV_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);

param.batch_size = 20;
param.step_size = 15*sqrt(param.batch_size);
[ipm_z20_iso, ipm_z20_iso_objective] = official_stochastic_iso_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);
[ipm_z20_AITV, ipm_z20_AITV_objective] = official_stochastic_AITV_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);

param.batch_size = 25;
param.step_size = 15*sqrt(param.batch_size);
[ipm_z25_iso, ipm_z25_iso_objective] = official_stochastic_iso_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);
[ipm_z25_AITV, ipm_z25_AITV_objective] = official_stochastic_AITV_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);


[ipm_zfull_AITV, ipm_zfull_AITV_objective] = AITV_ADMM_nonblind(N_obj, probe_true, dp1, ind_b, 0.15, 0.25, 0.25, 0.8, mode);
[ipm_zfull_iso, ipm_zfull_iso_objective] = iso_ADMM_nonblind(N_obj, probe_true, dp1, ind_b, 0.15, 0.25, 0.25, mode);
