function w = complexToReal(z)
% converts complex vector z to real-valued vector w of length 2N

    w = [real(z(:)); imag(z(:));];
    
end
