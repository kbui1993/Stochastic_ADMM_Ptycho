%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function computes the ssim between the phase of the reconstructed
%image and the phase of the ground truth.
%Input:
%   -z: reconstructed image
%   -obj_true: ground truth
%Output:
%   -result: ssim value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result = compute_angle_ssim(z, obj_true)
    result = ssim(angle(z), angle(exp(-1i*angle(trace(z'*obj_true)))*obj_true));
end