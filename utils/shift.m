%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function shifts each pixel of the probe by up to |px/dx_x| and 
%|py/dx_y| or equivalently performs phase rotation of each pixel of probe 
%in the range of [-px/dx_x, px/dx_x] and [-py/dx_y, py/dx_y].
%Input:
%   input: probe
%   lambda: wavelength
%   dx_x: pixel size (in sample plane) for the probe along x-axis
%   dx_y: pixel size (in sample plane) for the probe along y-axis
%   px: x position 
%   py: y position
%Output:
%   output: shifted probe
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [output] = shift( input, dx_x, dx_y, px, py )

Ny = size(input,1);
Nx = size(input,2);


% set up meshgrid
dk_y = 1/(dx_y*Ny);
dk_x = 1/(dx_x*Nx);
ky = linspace(-floor(Ny/2),ceil(Ny/2)-1,Ny);
kx = linspace(-floor(Nx/2),ceil(Nx/2)-1,Nx);
[kX,kY] = meshgrid(kx,ky);
kX = kX.*dk_x;
kY = kY.*dk_y;

% perform phase rotation on probe in fourier domain
f = fftshift(fft2(ifftshift(input)));
f = f.*exp(-2*pi*1i*px*kX).*exp(-2*pi*1i*py*kY);
output = fftshift(ifft2(ifftshift(f)));

end