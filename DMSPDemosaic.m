function res = DMSPDemosaic(observation, init, params)
% Implements stochastic gradient descent (SGD) Bayes risk minimization for image demosaicing described in:
% "Deep Mean-Shift Priors for Image Restoration" (http://home.inf.unibe.ch/~bigdeli/DMSPrior.html)
% S. A. Bigdeli, M. Jin, P. Favaro, M. Zwicker, Advances in Neural Information Processing Systems (NIPS), 2017
%
% Input:
% observation: mosaiced input image in range of [0, 255].
% init: Initialized demosaicing.
% params: Set of parameters.
% params.denoiser: The denoiser function hanlde.
%
% Optional parameters:
% params.sigma_dae: The standard deviation of the denoiser training noise. default: 11
% params.num_iter: Specifies number of iterations.
% params.mu: The momentum for SGD optimization. default: 0.9
% params.alpha: The step length in SGD optimization. default: 0.1
% params.gpu: Flag to run on GPU. default: false
%
% Outputs:
% res: Solution.
%
% Author:
% German F. Torres
%
% Note:
% This code was made based on deblurring implementation provided by
% authors' paper


if ~any(strcmp('denoiser',fieldnames(params)))
    error('Need a denoiser in params.denoiser!');
end

if ~any(strcmp('sigma_dae',fieldnames(params)))
    params.sigma_dae = 11;
end

if ~any(strcmp('num_iter',fieldnames(params)))
    params.num_iter = 300;
end

if ~any(strcmp('mu',fieldnames(params)))
    params.mu = .9;
end

if ~any(strcmp('alpha',fieldnames(params)))
    params.alpha = .1;
end

if ~any(strcmp('gpu',fieldnames(params)))
    params.gpu = false;
end

print_iter = any(strcmp('gt',fieldnames(params)));

sigma_di = 2.5;
sigma_d = 1;

pad = 0;
res = init;

step = zeros(size(res));

if print_iter
    psnr = computePSNR(params.gt, res, pad);
    disp(['Initialized with PSNR: ' num2str(psnr)]);
end

for iter = 1:params.num_iter
    if print_iter
        disp(['Running iteration: ' num2str(iter)]);
        tic();
    end
    
    % compute prior gradient
    input = res(:,:,[3,2,1]); % Switch channels for network    
    noise = randn(size(input)) * params.sigma_dae;
    
    rec = params.denoiser(input + noise);
        
    prior_grad = input - rec;
    prior_grad = prior_grad(:,:,[3,2,1]);
    
    % compute data gradient
    sampled_res = fwd_bayer(res, params.gpu);
    data_err = sampled_res-observation;
    data_grad = bwd_bayer(data_err, params.gpu);

    
    if iter==1
        %relative_weight = (1/sigma_di/sigma_di)/(1/sigma_di/sigma_di + 1/params.sigma_dae/params.sigma_dae);
        relative_weight = 1/(sigma_di^2);
    else
        %relative_weight = (1/sigma_d/sigma_d)/(1/sigma_d/sigma_d + 1/params.sigma_dae/params.sigma_dae);
        relative_weight = 1/(sigma_d^2);
    end
    
    % sum the gradients
    % grad_joint = data_grad*relative_weight + prior_grad*(1-relative_weight);
    grad_joint = data_grad*relative_weight + prior_grad*2/(params.sigma_dae^2);    
   
    % update
    step = params.mu * step - params.alpha * grad_joint;
    res = res + step;
    res = min(255,max(0,res));

    if print_iter
        psnr = computePSNR(params.gt, res, pad);
        disp(['PSNR is: ' num2str(psnr) ', iteration finished in ' num2str(toc()) ' seconds']);
    end
    
end
end

% Backward operator for the bayer filter
function out = bwd_bayer(x, use_gpu)
% x: bayer-paterned image
if use_gpu
    out = zeros(size(x,1), size(x,2), 3, 'gpuArray');
else
    out = zeros(size(x,1), size(x,2), 3);
end
% red channel
out(1:2:end, 1:2:end, 1) = x(1:2:end, 1:2:end);

% green channel
out(2:2:end, 1:2:end, 2) = x(2:2:end, 1:2:end);
out(1:2:end, 2:2:end, 2) = x(1:2:end, 2:2:end);

% blue channel
out(2:2:end, 2:2:end, 3) = x(2:2:end, 2:2:end);
end

% Forward operator for the bayer filter
function out = fwd_bayer(x, use_gpu)
% x: RGB image
if use_gpu
    out = zeros(size(x,1), size(x,2), 'gpuArray');
else
    out = zeros(size(x,1), size(x,2));
end
% red channel
out(1:2:end, 1:2:end) = x(1:2:end, 1:2:end, 1);

% green channel
out(2:2:end, 1:2:end) = x(2:2:end, 1:2:end, 2);
out(1:2:end, 2:2:end) = x(1:2:end, 2:2:end, 2);

% blue channel
out(2:2:end, 2:2:end) = x(2:2:end, 2:2:end, 3);
end
