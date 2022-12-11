%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function computes the signal-noise-ratio between the reconstructed
%image obtained by the blind ptychography problem and the ground-truth 
%image.
%Input:
%   z: reconstructed image
%   obj_true: ground truth image
%Output:
%   result: signal-noise-ratio value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [result,gamma] = compute_snr_blind(z,obj_true)

    %compute the optimal scaling factor
    gamma = sum(obj_true(:))/sum(z(:));
    
    %compute the relative squared error
    rel_err_square = norm(gamma*z-obj_true, 'fro')^2/norm(gamma*z, 'fro')^2;
    
    %compute snr
    result = -10*log10(rel_err_square);
end