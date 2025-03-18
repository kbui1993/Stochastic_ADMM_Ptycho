%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function generates ptychography data from two images. The
%ptychography scan is supposed to simulate velociprobe
%(https://www.osti.gov/servlets/purl/1565795).
%Input:
%   -mag_image: grayscale image to use as the magnitude image
%   -phase_image: grayscale image to use as the phase image
%   -N_scans: the maximum number of scans per direction, i.e. up to N_scans
%   along the x-axis and up to N-scans along the y_axis, so up to N_scans^2
%   total
%   -diffraction_size: size of the probe; must be less than size of image.
%Output:
%   -obj_true: overall scanned image; note that it may be smaller than the
%   original input images.
%   -probe_true: probe used to scan obj_true
%   -dp: set of magnitudes obtained by probing the obj_true
%   -ind_b: set of x,y indices for the masking matrix corresponding to dp
%   -dx: pixel size (in sample plane) for the probe
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [obj_true, probe_true, dp, ind_b, dx] = generate_ptycho_data(mag_image, phase_image, N_scans, diffraction_size)
    %% convert to double
    mag_image = double(mag_image);
    phase_image = double(phase_image);

    %% check input
    % check if images are square and mag_image and phase_image are same
    % sizes
    assert(size(mag_image,1)==size(phase_image,1));
    assert(size(mag_image,2)==size(phase_image,2));
    assert(size(mag_image,1)==size(mag_image,2));
    
    % check if diffraction size is less than image size
    assert(diffraction_size < size(phase_image,1));
    
    %% complex image and probe generation
    % generate complex image
    mag_image(mag_image==0) = 1;
    
    % normalize magnitude image to [0,1]
    mag_image = mag_image./max(mag_image(:));
    
    % normalize phase image to radian [0,1]
    phase_image = phase_image - min(phase_image(:));
    phase_image = phase_image./max(phase_image(:));
    obj_true = mag_image.*exp(1i*phase_image);

    % generate initial probe based on velociprobe microscope
    dx = 10e-9; %physical pixel size
    energy = 8.8;
    lambda = 1.23984193e-9/energy;
    Ls = diffraction_size*1e-6; %determine probe size
    probe_initial =  generate_probe(diffraction_size, lambda, dx, Ls, 'velo');

    %% generate scan positions
    % set parameters
    stepSize_true = 2.5e-7; %scan step size (can be adjusted)
    N_obj = size(obj_true,1); %only square object allowed for now
    ind_obj_center = floor(N_obj/2)+1;

    % create grid of positions and center them
    pos_x = (1 + (0:N_scans-1) *stepSize_true);
    pos_y = (1 + (0:N_scans-1) *stepSize_true);
    pos_x  = pos_x - (mean(pos_x));
    pos_y  = pos_y - (mean(pos_y));
    [Y,X] = meshgrid(pos_x, pos_y);
    ppX = X(:);
    ppY = Y(:);
    
    % avoid periodic artifacts, add some random offsets
    ppX = ppX + 10e-9*(rand(size(ppX))*2-1);
    ppY = ppY + 10e-9*(rand(size(ppY))*2-1);

    % calculate indicies for all scans
    % position = pi(integer) + pf(fraction)
    py_i = round(ppY/dx);
    py_f = ppY - py_i*dx;
    px_i = round(ppX/dx);
    px_f = ppX - px_i*dx;

    % compute minimum and maximum indices along x and y axises for probe
    % masks
    ind_x_lb = px_i - floor(diffraction_size/2) + ind_obj_center;
    ind_x_ub = px_i + ceil(diffraction_size/2) -1 + ind_obj_center;
    ind_y_lb = py_i - floor(diffraction_size/2) + ind_obj_center;
    ind_y_ub = py_i + ceil(diffraction_size/2) -1 + ind_obj_center;
    
    % eliminate out-of-bound indices
    valid_ind_x_lb = (ind_x_lb > 0);
    valid_ind_x_ub = (ind_x_ub <= N_obj);
    valid_ind_y_lb = (ind_y_lb > 0);
    valid_ind_y_ub = (ind_y_ub <= N_obj);
    ind_x_lb = ind_x_lb(valid_ind_x_lb & valid_ind_x_ub & valid_ind_y_lb & valid_ind_y_ub);
    ind_x_ub = ind_x_ub(valid_ind_x_lb & valid_ind_x_ub & valid_ind_y_lb & valid_ind_y_ub);
    ind_y_lb = ind_y_lb(valid_ind_x_lb & valid_ind_x_ub & valid_ind_y_lb & valid_ind_y_ub);
    ind_y_ub = ind_y_ub(valid_ind_x_lb & valid_ind_x_ub & valid_ind_y_lb & valid_ind_y_ub);
    
    % crop image and adjust position
    min_x = min(ind_x_lb);
    max_x = max(ind_x_ub);
    min_y = min(ind_y_lb);
    max_y = max(ind_y_ub);
    
    obj_true = obj_true(min_y:max_y, min_x:max_x);
    ind_x_lb = ind_x_lb - min_x+1;
    ind_x_ub = ind_x_ub - min_x+1;
    ind_y_lb = ind_y_lb - min_y+1;
    ind_y_ub = ind_y_ub - min_y+1;
    
    % aggregate all the positions into one matrix
    ind_b=[ind_x_lb,ind_x_ub,ind_y_lb,ind_y_ub];

    % overlap ratio between scans
    overlapRatio = 1-stepSize_true/dx/N_scans^2;
    overlapRatio

    % shift initial probe to create true probe
    probe_true = shift(probe_initial, dx, dx, px_f(1), py_f(1));
    
    %% compute the scans
    N_scan = length(ind_x_lb);
    dp = zeros(diffraction_size,diffraction_size,N_scan);
    for i=1:N_scan
        % scan the object 
        obj_roi = obj_true(ind_y_lb(i):ind_y_ub(i),ind_x_lb(i):ind_x_ub(i));
        psi =  obj_roi .* probe_true;
    
        %FFT to get diffraction pattern
        dp(:,:,i) = abs(fftshift(fft2(ifftshift(psi)))).^2;
    end
end