%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function performs a SGD version of ADMM
%to solve the nonblind ptychography problem with isotropic TV.
%Input:
%   N_obj: length of square image
%   probe: ground-truth probe
%   dp: set of magnitudes
%   ind_b: set of x,y indices for the masking matrix
%   lambda: regularization parameter for AITV term
%   beta1: penalty parameter for uj - F(P_jz)
%   beta2: penalty parameter for v - Dz
%   batch_size: number of scans to access per batch
%   step_size: step size parameter for stochastic gradient descent
%   mode:
%       'AGM' - amplitude Gaussian metric
%       'IPM' - intensity based Poisson metric
%Output:
%   z: output image
%   objective: vector of the objective values where index corresponds to
%              iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z, objective, best_z] = official_stochastic_AITV_blind_ADMM(N_obj, dp, ind_b, param, mode)

    %% set default parameters
    if isfield(param, 'lambda') == 0
        param.lambda = 10;
    end

    if isfield(param, 'beta1') == 0
        param.beta1 = 0.25;
    end
    
    if isfield(param, 'beta2') == 0
        param.beta1 = 0.25;
    end

    if isfield(param, 'batch_size') == 0
        param.batch_size = 20;
    end

    if isfield(param, 'z_update_mode') == 0
        param.z_update_mode = "";
    elseif strcmp(param.z_update_mode, 'PIE') || strcmp(param.z_update_mode, 'ePIE') || strcmp(param.z_update_mode, 'rPIE')
        if isfield(param, 'gamma') == 0
            param.gamma = 0.10;
        end
    else
        param.z_update_mode = "rPIE";
        param.gamma = 0.10;
    end

    if isfield(param, 'step_size') == 0 && strcmp(param.z_update_mode, "rPIE")
        param.step_size = 2*sqrt(param.batch_size);
    elseif strcmp(param.z_update_mode, "")
        param.step_size = 0.0001;
    end

    if isfield(param, 'omega_update_mode') == 0
        param.z_update_mode = "";
    elseif strcmp(param.omega_update_mode, 'PIE') || strcmp(param.omega_update_mode, 'ePIE') || strcmp(param.omega_update_mode, 'rPIE')
        if isfield(param, 'gamma_omega') == 0
            param.gamma_omega = 0.10;
        end
    else
        param.omega_update_mode = "rPIE";
        param.gamma_omega = 0.10;
    end

    
    %% obtain image data
    %obtain size of probe
    m = size(param.omega,1);

    %obtain number of masks
    n = size(ind_b,1);

    %compute number of inner iterations for each epoch
    inner_iter = ceil(n/param.batch_size);
    
    %compute the number of scans per pixel
    full_count_mat = zeros(N_obj, N_obj);
    
    for i = 1:n
        full_count_mat(ind_b(i,3):ind_b(i,4), ind_b(i,1):ind_b(i,2)) = ...
                        full_count_mat(ind_b(i,3):ind_b(i,4), ind_b(i,1):ind_b(i,2)) + 1;
    end
    
    %for pixels with zero scans, set the number of scans to 1 to avoid
    %division by zero
    full_count_mat(full_count_mat == 0) = 1;

    %% preinitialize variables
    if isfield(param, 'init_z') == 0
        z = (ones(N_obj, N_obj)+1i*ones(N_obj, N_obj))/sqrt(2);
    else
        z = param.init_z;
    end
    u = ones(m,m, n)+1i*ones(m,m, n);
    Lambda = u;
    yx = Dx(z);
    yy = Dy(z);



    %% set parameters
    step_size = param.step_size;
    beta1 = param.beta1;
    beta2 = param.beta2;
    lambda = param.lambda;
    batch_size = param.batch_size;
    omega = param.omega;
    alpha = param.alpha;

    %% preinitialize
    objective = zeros(param.maxiter,1);

    %run stochastic ADMM
    for i = 1:param.maxiter

        %store old z
        z_old = z;

        %randomly permute the scans
        updateOrder = randperm(n);

        %run an epoch 
        for j=1:inner_iter

            %% get the batch
            start_index = batch_size*(j-1)+1;
            end_index = min(batch_size*j, n);
            batch_updateOrder = updateOrder(start_index:end_index);

            %% solve u_j subproblem

            %also count the number of scans per pixel within a batch
            count_mat = zeros(N_obj, N_obj);
            
            %fidelity term if AGM
            if strcmp(mode, 'AGM')
                for k = 1:size(batch_updateOrder,2)
                    new_idx = batch_updateOrder(k);
                    u(:,:,new_idx) = prox_AGM(z(ind_b(new_idx,3):ind_b(new_idx,4), ind_b(new_idx,1):ind_b(new_idx,2)),...
                        omega, dp(:,:,new_idx), Lambda(:,:,new_idx), beta1);
                    count_mat(ind_b(new_idx,3):ind_b(new_idx,4), ind_b(new_idx,1):ind_b(new_idx,2)) = ...
                        count_mat(ind_b(new_idx,3):ind_b(new_idx,4), ind_b(new_idx,1):ind_b(new_idx,2)) + 1;
                end
            %otherwise fidelity term is IPM
            else
                for k = 1:size(batch_updateOrder,2)
                    new_idx = batch_updateOrder(k);
                    u(:,:,new_idx) = prox_IPM(z(ind_b(new_idx,3):ind_b(new_idx,4), ind_b(new_idx,1):ind_b(new_idx,2)), omega, dp(:,:,new_idx), Lambda(:,:,new_idx), beta1);
                    count_mat(ind_b(new_idx,3):ind_b(new_idx,4), ind_b(new_idx,1):ind_b(new_idx,2)) = ...
                        count_mat(ind_b(new_idx,3):ind_b(new_idx,4), ind_b(new_idx,1):ind_b(new_idx,2)) + 1;
                end
            end
            
            %for pixels with zero scans, set the number of scans to 1 to avoid
            %division by zero
            count_mat(count_mat==0) = 1;


            %% omega subproblem 
            omega_grad = zeros(size(omega));
            for k=1:size(batch_updateOrder,2)
                new_idx = batch_updateOrder(k);
                z_mask = z(ind_b(new_idx,3):ind_b(new_idx,4), ind_b(new_idx,1):ind_b(new_idx,2));

                %% set PIE step size
                if strcmp(param.omega_update_mode, 'ePIE')
                    omega_PIE_step_size = 1/max(max(abs(z_mask).^2));
                elseif strcmp(param.omega_update_mode, 'rPIE')
                    peak = max(max(abs(z_mask).^2));
                    omega_PIE_step_size=1./((1-param.gamma_omega)*abs(z_mask).^2+param.gamma_omega*peak);
                elseif strcmp(param.omega_update_mode, 'PIE')
                    peak = max(max(abs(z_mask).^2));
                    omega_PIE_step_size = abs(z_mask)./(peak*(abs(z_mask).^2+param.gamma*peak^2));
                else
                    omega_PIE_step_size = 1;
                end

                omega_grad = omega_grad+omega_PIE_step_size.*conj(z_mask).*(omega.*z_mask - fftshift(ifft2(ifftshift(u(:,:,new_idx)+Lambda(:,:,new_idx)/beta1))));
            end
            
            omega = omega -param.probe_step_size*(beta1/size(batch_updateOrder,2))*omega_grad;


           %% solve the v-subproblem

            %Dz - y/beta2 
            temp1 = Dx(z)-yx/beta2;
            temp2 = Dy(z)-yy/beta2;
            
            temp = shrinkL12([temp1(:),temp2(:)], lambda/beta2, alpha);
            
            %separate
            vx = reshape(temp(:,1), N_obj, N_obj);
            vy = reshape(temp(:,2), N_obj, N_obj);

            %% solve z-subproblem by stochastic gradient descent

            %compute the gradient term of the quadratic fidelity term for
            %Dx(z) and Dy(z): (beta_2/2)*\|v^t-Dz +y^t/beta_2\|^2
            tv_term = beta2*(Dxt(vx+yx/beta2-Dx(z))+...
                Dyt(vy+yy/beta2-Dy(z)));
            tv_term = tv_term./full_count_mat;
            
            %compute the full stochastic batch gradient

            %% set PIE step size
            if strcmp(param.z_update_mode, 'ePIE')
                PIE_step_size = 1/max(max(abs(omega).^2));
            elseif strcmp(param.z_update_mode, 'rPIE')
                peak = max(max(abs(omega).^2));
                PIE_step_size=1./((1-param.gamma)*abs(omega).^2+param.gamma*peak);
            elseif strcmp(param.z_update_mode, 'PIE')
                peak = max(max(abs(omega).^2));
                PIE_step_size = abs(omega)./(peak*(abs(omega).^2+param.gamma*peak^2));
            else
                PIE_step_size = 1;
            end

            sum_term = zeros(N_obj, N_obj);
            z = reshape(z, N_obj, N_obj);
            for k = 1:size(batch_updateOrder,2)
                new_idx = batch_updateOrder(k);
                obj_roi = z(ind_b(new_idx,3):ind_b(new_idx,4), ind_b(new_idx,1):ind_b(new_idx,2));
                new_obj_roi = conj(omega).*omega.*obj_roi;
                
                %compute  the gradient of the  quadratic fidelity term for F(P_jz): 
                %(\beta_1/2)*\|u_j^t - F(P_jz) + w_j^t/beta_1\|^2
                update_term = conj(omega).*fftshift(ifft2(ifftshift(u(:,:,new_idx)+Lambda(:,:,new_idx)/beta1)))-...
                    new_obj_roi;
                
                %add the gradient term batch by batch
                sum_term(ind_b(new_idx,3):ind_b(new_idx,4), ind_b(new_idx,1):ind_b(new_idx,2)) = ...
                    sum_term(ind_b(new_idx,3):ind_b(new_idx,4), ind_b(new_idx,1):ind_b(new_idx,2)) - PIE_step_size.*(beta1*update_term + ...
                    tv_term(ind_b(new_idx,3):ind_b(new_idx,4), ind_b(new_idx,1):ind_b(new_idx,2)));
            end

            %gradient descent step
            z = z - step_size*(sum_term./count_mat);
            
            %% update Lagrange multipliers
            for k = 1:size(batch_updateOrder,2)
                new_idx = batch_updateOrder(k);
                obj_roi = z(ind_b(new_idx,3):ind_b(new_idx,4), ind_b(new_idx,1):ind_b(new_idx,2));
                psi = omega.*obj_roi;
                F_psi = fftshift(fft2(ifftshift(psi)));
                Lambda(:,:,new_idx) = Lambda(:,:,new_idx)+beta1*(u(:,:,new_idx)-F_psi);
            end
            
            yx = yx+beta2*(vx-Dx(z));
            yy = yy+beta2*(vy-Dy(z));
        end

        %compute relative error
        err = norm(abs(z)-abs(z_old), 'fro')/norm(abs(z_old), 'fro');
        
        %print iteration and error
        if mod(i,10)==0
            disp(['iterations: ' num2str(i) '!  ' 'error is:   ' num2str(err)]);
        end

        %decrease step size by a factor of 10
        if i==floor(param.maxiter)*0.5 || i==floor(param.maxiter)*0.75 
            step_size = step_size*0.1;
            param.probe_step_size = param.probe_step_size*0.1;
        end
        
        %compute objective value
        objective(i) = obj_value(z, omega, ind_b, dp, alpha, mode);
        
        if i == 1
            best_obj = objective(1);
            best_z = z;
        end
        
        if objective(i) < best_obj
            best_obj = objective(i);
            best_z = z;
        end

    end



end

function d = Dx(u)
    %obtain matrix size of u
    [rows,cols] = size(u);
    
    %preinitialize matrix
    d = zeros(rows,cols);
    
    %compute Dx(u)
    d(:,2:cols) = u(:,2:cols)-u(:,1:cols-1);
    d(:,1) = u(:,1)-u(:,cols);
end

function d = Dxt(u)
    
    %obtain size of u
    [rows,cols] = size(u);
    
    %preinitialize
    d = zeros(rows,cols);
    
    %compute Dx^t(u)
    d(:,1:cols-1) = u(:,1:cols-1)-u(:,2:cols);
    d(:,cols) = u(:,cols)-u(:,1);
end

function d = Dy(u)
    %obtain size of matrix
    [rows,cols] = size(u);
    
    %preinitialize
    d = zeros(rows,cols);
    
    %compute Dy(u)
    d(2:rows,:) = u(2:rows,:)-u(1:rows-1,:);
    d(1,:) = u(1,:)-u(rows,:);
end

function d = Dyt(u)

    %obtain size of u
    [rows,cols] = size(u);
    
    %preinitialize
    d = zeros(rows,cols);
    
    %compute Dy^t(u)
    d(1:rows-1,:) = u(1:rows-1,:)-u(2:rows,:);
    d(rows,:) = u(rows,:)-u(1,:);
end



function uj = prox_AGM(image, probe, dp_j, Lambdaj, beta1)
    %this function is the proximal operator for AGM
    psi = probe.*image;
    F_psi = fftshift(fft2(ifftshift(psi)));
    F_p_w = F_psi-Lambdaj/beta1;
    uj = sign(F_p_w).*(sqrt(dp_j)+beta1*abs(F_p_w))/(1+beta1);
end

function uj = prox_IPM(image, probe, dp_j, Lambdaj, beta1)
    %this function is the proximal operator for IPM
    psi = probe.*image;
    F_psi = fftshift(fft2(ifftshift(psi)));
    F_p_w = F_psi-Lambdaj/beta1;
    uj = (sign(F_p_w)/(2*(1+beta1))).*(beta1*abs(F_p_w)+sqrt((beta1*abs(F_p_w)).^2+4*(1+beta1)*dp_j));
end


function x = shrinkL12(y,lambda,alpha)
    %this function applies the proximal operator of L1-alpha L2 to each
    %row vector

    x = zeros(size(y));

    [max_y, idx_y] = max(abs(y'));
    max_y = max_y';
    idx_y = idx_y';
    new_idx_y = sub2ind(size(y), (1:size(y,1))',idx_y);
    
    case1_idx = max_y > lambda;
    
    case1_result = max(abs(y(case1_idx,:))-lambda,0).*sign(y(case1_idx,:));
    norm_case1_result = sqrt(sum(case1_result.^2,2));
    x(case1_idx,:) =((norm_case1_result+alpha*lambda)./norm_case1_result).*case1_result;
    
    case2_idx = logical((max_y<=lambda).*(max_y>=(1-alpha)*lambda));
    
    x(new_idx_y(case2_idx)) = (max_y(case2_idx)+(alpha-1)*lambda).*sign(y(new_idx_y(case2_idx)));
    
end


function val = obj_value(z, probe, ind_b, dp, alpha, mode)
    %get number of masks
    n = size(ind_b,1);
    
    %preinitialize
    val = 0;
    
    %compute
    for i = 1:n
        
        %get magnitude of reconstructed image
        psi = probe.*(z(ind_b(i,3):ind_b(i,4), ind_b(i,1):ind_b(i,2)));
        F_psi = fftshift(fft2(ifftshift(psi)));
        
        %compute the metric
        if strcmp(mode, 'AGM')
            diff_val = 0.5*norm(abs(F_psi)-sqrt(dp(:,:,i)),'fro');
        else
            Poisson_dif = 0.5*(abs(F_psi).^2-dp(:,:,i).*log(abs(F_psi).^2));
            diff_val = sum(Poisson_dif(:));
        end
        
        %add it up
        val = val + diff_val;
    end
    
%     %compute the total variation term
%     Dx_val = Dx(z);
%     Dy_val = Dy(z);
%     
%     TV_val = abs(Dx_val(:))+abs(Dy_val(:)) - alpha*(sqrt(abs(Dx_val(:)).^2 + abs(Dy_val(:)).^2));
%     
%     %add it up
%     val = val + sum(TV_val(:));
end