% load class_labels
load('/Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myCode/class_name.mat');
class_cell = data';
clear data class

% load fisher vector
[filename,pathname] = uigetfile('*.mat','Load Fisher Vector');
load(fullfile(pathname,filename));
FV = FV';

supervised_learn(FV, class_cell)
