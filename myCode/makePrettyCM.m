cm_svm = load('/Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myData/fisherVectors/numClusters10_exemplarSize3/svm_cm_total_test.mat');
total_confusion_matrix = cm_svm.CM_SVM;
f = figure;
imagesc(total_confusion_matrix/10)
ax = gca;
ax.FontSize = 20;
ax.XTickLabel = unique(testing_classes);
ax.YTickLabel = unique(testing_classes);
colorbar;
set(f, 'Position', [100, 100, 1049, 895]);
%%
cm_svm1 = load('/Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myData/fisherVectors/numClusters10_exemplarSize3/cm_svm2.mat');
cm_svm2 = load('/Users/sagarsetru/Documents/Princeton/cos424/hw1/voxResources/myData/fisherVectors/numClusters10_exemplarSize3/cm_svm_total.mat');
total_confusion_matrix1 = cm_svm1.CM_SVM;
total_confusion_matrix2 = cm_svm2.CM_SVM;
total_confusion_matrix = total_confusion_matrix2-total_confusion_matrix1;
f = figure;
imagesc(total_confusion_matrix/10)
colormap(redblue)
ax = gca;
ax.FontSize = 20;
ax.XTickLabel = unique(testing_classes);
ax.YTickLabel = unique(testing_classes);
colorbar;
set(f, 'Position', [100, 100, 1049, 895]);
%%
f2 = figure;
plot(diag(total_confusion_matrix),'b.')
ax = gca;
ax.FontSize = 20;
ylim([0 100])
ax.XTickLabel = unique(testing_classes);
