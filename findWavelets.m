function [amplitudes,f] = findWavelets(projections,H,K,fs)
%findWavelets finds the wavelet transforms resulting from a time series
% (C) Gordon J. Berman, 2014
%     Modified by Mochi Liu, 2016
%     Princeton University
    

    omega0 = 5;
    dt = 1 ./ fs;
    minF = 0;
    maxF = fs/2; %nyquist frequency
    f = linspace(minF, maxF, K);
    
    N = length(projections(:,1));
    amplitudes = zeros(N,K);
    
    %find the relevant frequencies
    relevant_f = find(sum(H,1));

    amplitudes(:,relevant_f) = ...
        fastWavelet_morlet_convolution_parallel(...
        projections,f(relevant_f),omega0,dt)';
    
    amplitudes = amplitudes';

    
    
%     if parameters.numProcessors > 1 && parameters.closeMatPool
%         matlabpool close
%     end
    
    
    
    
    