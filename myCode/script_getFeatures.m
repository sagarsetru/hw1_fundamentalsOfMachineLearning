%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FV Concatenating
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change this to the directory containing your data folder
%dirn = '/Users/Gazelle/Documents/voxDemo/';
dirn = '/Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/';

% add tools path
addpath(genpath('/Users/sagarsetru/Documents/Princeton/cos424/hw1/voxDemo/'))
addpath(genpath('/Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myCode'))

% use variable name 'class' instead of 'LB' for use with function demo_fv.m 
[DAT, class, FNS] = loadAll(dirn);

%% extract fields from all structures
cd /Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myData/featureStructures;
% get names of fields
fields = fieldnames(DAT{1}); 
% loop over field names
for field = fields', 
    disp(['Processing field: ',char(field)]);
    % define cell to store features from each song
    % use variable name 'data' for use with demo_fv.m
    data = cell(1,length(DAT));
    % loop over songs
    for i = 1:length(DAT),
        % load structure for each song
        D = DAT{i};
        % store current field from each song in data
        data{i} = D.(char(field));
    end
    save(field{:},'data','class');
end

%% load feature structures
cd /Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myData/featureStructures;
% pick up .mat files
featFiles = uipickfiles;
%%
cd /Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myData/fisherVectors;
% set numClusters
numClusters = 10;
% set examplarSizes
exemplarSizes = 5;
% fails for 1:  
% inharmonic

% fails for 3, 5:
% class
% inharmonic
% key
% tempo


% loop over files
for file = featFiles,
    gendata = load(file{:});
    % get the name of the current feature
    [pathName,featName] = fileparts(file{:});
    disp(['Generating Fisher vector for: ',featName])
    for numCluster = numClusters,
        for exemplarSize = exemplarSizes,
        % set directory name
        dirName = strcat('numClusters',num2str(numClusters),'_','exemplarSize',num2str(exemplarSize));
        if ~exist(dirName),
            mkdir(dirName);
        end
        currentDirectory = pwd;
        [upperPath, deepestFolder, ~] = fileparts(currentDirectory);
        if ~strcmp(deepestFolder,dirName)
            cd(dirName);
        end
        % generate fisher vector
        %FV = demo_fv(gendata, numCluster, examplarSize);
        FV = demo_fv(gendata);%, numCluster, examplarSize);

        save(featName,'FV');
        end 
    end
end
LB = gendata.class;
save('LB.mat','LB');