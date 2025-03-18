%%clear workspace
%clear all;

%turn off more
more off;

%%%%%%%%%%%%%%%%%%%%%%%%%%load data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fprintf('Load data...\n');
% sample = 'chip';
sample = 'cameraman';
stepSize_true = 1e-07;
datastr=[sample,num2str(stepSize_true),'_100_aperiodic'];
load([datastr,'.mat'], '-mat');


%The data the contains the following variables
%   -obj: complex image
%   -dx: step size used for translating probe
%   -dp: the set of scans obtained by probing the object.
%   -ppX: set of x-coordinates of the probe
%   -ppY: set of y-coordinates of the probe
%   -probe_true: true_probe

%%%%%%%%%%%%%%%%%%%%%%Extract size of probe and object%%%%%%%%%%%%%%%%%%%%%
%get indices of the probes
subind=1:size(dp,3);

%set rng
rng(1234)

%get size of probe
Np = size(dp,1);

%obtain the center x,y-coordinates of the probes
px = ppX(subind);
px = px(:);
py = ppY(subind);
py = py(:);

%get max location of x,y-coordinate with respect to the image
Nx_max = max(abs(round(min(px)/dx)-floor(Np/2)), abs(round(max(px)/dx)+ceil(Np/2)))*2+1;
Ny_max = max(abs(round(min(py)/dx)-floor(Np/2)), abs(round(max(py)/dx)+ceil(Np/2)))*2+1;

%get size of true image
N_obj = size(obj,1);

%obtain center location of image
ind_obj_center = floor(N_obj/2)+1;

%get number of scans
N_scan = length(px);

%rescale probe center x,y-coordinate to image x,y-cordinate relative to the center of
%the image
px_i = round(px/dx);
py_i = round(py/dx);

%compute lower and upper bound x,y-coordinates of probes
ind_x_lb = px_i - floor(Np/2) + ind_obj_center;
ind_x_ub = px_i + ceil(Np/2) -1 + ind_obj_center;
ind_y_lb = py_i - floor(Np/2) + ind_obj_center;
ind_y_ub = py_i + ceil(Np/2) -1 + ind_obj_center;
ind_b=[ind_x_lb,ind_x_ub,ind_y_lb,ind_y_ub];


%%%%%%%%%%%%%%%%%%%%%%%generate ground-truth%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%create copy of true complex image
obj_true = obj;

%convert to real
v_obj_true = complexToReal(obj_true(:));
v_p_true = complexToReal(probe_true(:));

%concatenate
v_true = [v_obj_true; v_p_true];

%%%%%%%%%%%%%%%%%%%generate initial probe using disk%%%%%%%%%%%%%%%%%%%%%%%
%set of constants
energy = 8.8;
lambda = 1.23984193e-9/energy;
dia_pert=1;

%generate probe
[probe0] =  generate_probe(Np, lambda, dx, Np*1e-6*dia_pert, 'velo');

%%%%%%%%%%%%%%%%%%intialize object%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialization setting
init_o ='ones';
% init_o = 'rando';

%initialize object
if(strcmp(init_o,'rando'))
    v_obj0=rand(N_obj,N_obj)+1i*rand(N_obj,N_obj);
else
    v_obj0=1.*(ones(N_obj,N_obj)+1i*ones(N_obj,N_obj));
end

%%%%%%%%%%%%%%Preprocess object and probe initialization%%%%%%%%%%%%%%%%%%%
%normalize and convert to real for object initialization
v_obj0 = complexToReal(v_obj0(:)./abs(v_obj0(:)));

%convert to complex
v_p0 = complexToReal(probe0(:));

%concatenate
v0 = [v_obj0;v_p0];
