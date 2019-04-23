clc, clear, close all;
addpath(genpath('DAEs'))

% select denoiser
% denoiser_name = 'caffe'; % make sure matCaffe is installed and its location is added to path
denoiser_name = 'matconvnet'; % make sure matconvnet is installed and its location is added to path

% set to 0 if you want to run on CPU (very slow)
use_gpu = 1;
% use_gpu = 0;

%% Load data

% Input file
file = '190.png';       % Check available images in folder data

% Read groundtruth
gt = double(im2uint8(imread(['data\groundtruth\' file])));
% Read input (mosaiced image)
input = double(im2uint8(imread(['data\input\' file])));
% Initialization
init = double(im2uint8(demosaic(uint8(input), 'rggb')));

if use_gpu
    gt = gpuArray(gt);
    input = gpuArray(input);
    init = gpuArray(init);
end

% load denoiser for solver
params.denoiser = loadDenoiser(denoiser_name, use_gpu, size(gt));

% set parameters
params.sigma_dae = 11; % correspones to the denoiser's training standard deviation
params.num_iter = 300; % number of iterations
params.gt = gt; % to print PSNR at each iteration
params.gpu = logical(use_gpu); % to run on GPU


%% demosaicing demo

% run DMSP demosaicing
restored = DMSPDemosaic(input, init, params);

figure;
subplot(131);
imshow(gt/255); title('Ground Truth')
subplot(132);
imshow(init/255); title('matlab demosaic')
subplot(133);
imshow(restored/255); title('Restored')