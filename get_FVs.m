%This script extracts the Fisher Vectors

parentDir = 'C:\Users\mochil\Downloads\genres\';

% Get a list of all files and folders in this folder.
files = dir(parentDir);
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
subFolders = files(dirFlags);
subFolders(1:2) = [];
% Print folder names to command window.
GENDATA.data = cell(1,1000);
song_index = 1;
for subFolder_index = 1:length(subFolders)
    curDir = [parentDir, subFolders(subFolder_index).name, '\'];
    files = dir([curDir, 'MFCCs_*']);
    dirFlags = [files.isdir];
    files = files(~dirFlags);
    
    for file_index = 1:length(files)
        file_name = files(file_index).name;
        load([curDir, file_name],'-mat');
        GENDATA.data{song_index} = MFCCs;
        song_index = song_index + 1;
    end
    
end

%get rid of NaNs
for cell_index = 1:1000
    nan_indecies = isnan(GENDATA.data{cell_index});
    GENDATA.data{cell_index}(nan_indecies) = 0;
end

FV = demo_fv(GENDATA, 10, 5);

