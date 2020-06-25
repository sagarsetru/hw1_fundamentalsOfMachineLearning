%This script extracts the Wavelet_MFCC and MFCC feature vectors

Tw = 25;           % analysis frame duration (ms)
Ts = 10;           % analysis frame shift (ms)
alpha = 0.97;      % preemphasis coefficient
R = [ 300 3700 ];  % frequency range to consider
M = 20;            % number of filterbank channels 
C = 13;            % number of cepstral coefficients
L = 22;            % cepstral sine lifter parameter

% hamming window (see Eq. (5.2) on p.73 of [1])
hamming = @(N)(0.54-0.46*cos(2*pi*[0:N-1].'/(N-1)));

parentDir = 'C:\Users\mochil\Downloads\genres\';

% Get a list of all files and folders in this folder.
files = dir(parentDir);
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
subFolders = files(dirFlags);
subFolders(1:2) = [];
% Print folder names to command window.
for subFolder_index = 1:1
    curDir = [parentDir, subFolders(subFolder_index).name, '\'];
    files = dir([curDir, '*.au']);
    dirFlags = [files.isdir];
    files = files(~dirFlags);
    
    for file_index = 1:1
        file_name = files(file_index).name;
        no_extension_name = file_name(1:end-3);
        % Read speech samples, sampling rate and precision from file
        [ speech, fs] = audioread([curDir, file_name]);

        % Feature extraction (feature vectors as columns)
        [ wavelet_MFCCs, ~ ] = wavelet_mfcc( speech, fs, Tw, Ts, alpha, R, M, C, L );

        Nw = round(1E-3*Tw*fs);
        time_bin_number = floor(size(wavelet_MFCCs, 2)/Nw);
        wavelet_MFCCs_small = wavelet_MFCCs(:,1:time_bin_number*Nw); %truncate so we can bin
        wavelet_MFCCs_small = reshape(wavelet_MFCCs_small,size(wavelet_MFCCs,1),[],Nw);
        wavelet_MFCCs_small = squeeze(sum(wavelet_MFCCs_small,3)) ./ Nw;

        % Feature extraction (feature vectors as columns)
        [ MFCCs, FBEs ] = ...
                      mfcc( speech, fs, Tw, Ts, alpha, hamming, R, M, C, L );
        save([curDir, 'wavelet_', no_extension_name], 'wavelet_MFCCs');
        save([curDir, 'MFCCs_', no_extension_name], 'MFCCs');
        save([curDir, 'smallwavelet_', no_extension_name], 'wavelet_MFCCs_small');
    end
    
end
          old_MFCC = data{1};
          % Plot cepstrum over time
          figure('Position', [30 100 1600 400], 'PaperPositionMode', 'auto', ... 
                 'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 
      
          imagesc( [1:size(old_MFCC,2)], [0:size(old_MFCC,1)], old_MFCC ); 
          axis( 'xy' );
          xlabel( 'Frame index' ); 
          ylabel( 'Cepstrum index' );
%           title( 'Provided DFT Mel frequency cepstrum' );
        set(gca,'FontSize',20)
            
          % Plot cepstrum over time
          figure('Position', [30 100 1600 400], 'PaperPositionMode', 'auto', ... 
                 'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 
      
          imagesc( [1:size(MFCCs,2)], [0:C-1], MFCCs ); 
          axis( 'xy' );
          xlabel( 'Frame index' ); 
          ylabel( 'Cepstrum index' );
%           title( 'HTK MFCC Mel frequency cepstrum' );
        set(gca,'FontSize',20)
          
          % Plot cepstrum over time
          figure('Position', [30 100 1600 400], 'PaperPositionMode', 'auto', ... 
                 'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 
      
          imagesc( [1:size(wavelet_MFCCs_small,2)], [0:C-1], wavelet_MFCCs_small ); 
          axis( 'xy' );
          xlabel( 'Frame index' ); 
          ylabel( 'Cepstrum index' );
%           title( 'Binned Wavlet Mel frequency cepstrum' );
        set(gca,'FontSize',20)
          
          % Plot cepstrum over time
          figure('Position', [30 100 1600 400], 'PaperPositionMode', 'auto', ... 
                 'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 
      
          imagesc( [1:size(wavelet_MFCCs,2)], [0:C-1], wavelet_MFCCs ); 
          axis( 'xy' );
          xlabel( 'Time point' ); 
          ylabel( 'Cepstrum index' );
%           title( 'Full Wavlet Mel frequency cepstrum' );
        set(gca,'FontSize',20)
