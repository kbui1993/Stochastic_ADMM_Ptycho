%setup
setup;
%setup_chip2;

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
param.gamma = 0.1;
param.maxiter = 600;
param.alpha = 0.8;

rng('shuffle')

DR_z = nonblind_ptycho_DR(N_obj, sqrt(dp1), ind_b, probe_true, 20);

rng('shuffle')

[agm_z10_iso, agm_z10_iso_objective] = official_stochastic_iso_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);
[agm_z10_AITV, agm_z10_AITV_objective] = official_stochastic_AITV_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);

param.batch_size = 20;
param.step_size = 2*sqrt(param.batch_size);
[agm_z20_iso, agm_z20_iso_objective] = official_stochastic_iso_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);
[agm_z20_AITV, agm_z20_AITV_objective] = official_stochastic_AITV_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);

param.batch_size = 50;
param.step_size = 2*sqrt(param.batch_size);
[agm_z50_iso, agm_z50_iso_objective] = official_stochastic_iso_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);
[agm_z50_AITV, agm_z50_AITV_objective] = official_stochastic_AITV_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);

param.batch_size = 5;
param.step_size = 2*sqrt(param.batch_size);
[agm_z5_iso, agm_z5_iso_objective] = official_stochastic_iso_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);
[agm_z5_AITV, agm_z5_AITV_objective] = official_stochastic_AITV_nonblind_ADMM(N_obj, probe_true, dp1, ind_b, param, mode);

[agm_zfull_AITV, agm_zfull_AITV_objective] = AITV_ADMM_nonblind(N_obj, probe_true, dp1, ind_b, 10, 0.25, 0.25, 0.8, mode);
[agm_zfull_iso, agm_zfull_iso_objective] = iso_ADMM_nonblind(N_obj, probe_true, dp1, ind_b, 10, 0.25, 0.25, mode);