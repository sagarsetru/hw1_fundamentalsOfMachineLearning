function [correct_ratio, total_confusion_matrix] = supervised_learn(FV, class_cell, method)
    
    %input fisher vector FV (N data points X A dimensional fisher vector)
    %input class_cell (N data points X 1) cell array
    % cross-validation: train on 90% of data. test on 10%. iterate 10 times
    
    if nargin < 3
        method = 'naive bayes';
    end
    
    training_ratio = 0.9;
    cross_validation_iteration = 10;
    dataset_length = length(class_cell);
    training_data_point_number = round(dataset_length*training_ratio);

    % FV = FV';
    % class_cell = class_cell';
    % 
    total_confusion_matrix = [];
    for iteration = 1:cross_validation_iteration
        %randomly select the training set
        random_indecies = randperm(dataset_length);
        training_indecies = random_indecies(1:training_data_point_number);
        testing_indecies = random_indecies(training_data_point_number+1:end);
        training_classes = class_cell(training_indecies);
        training_FV = FV(training_indecies,:);
        testing_classes = class_cell(testing_indecies);
        testing_FV = FV(testing_indecies,:);

        machine_learning_object = fitNaiveBayes(training_FV,training_classes);

%        Predicted_result = machine_learning_object.predict(testing_FV);
        Predicted_result = machine_learning_object.predict(testing_FV);
        confusion_matrix = confusionmat(testing_classes,Predicted_result);
        if isempty(total_confusion_matrix)
            total_confusion_matrix = confusion_matrix;
        else
            total_confusion_matrix = total_confusion_matrix + confusion_matrix;
        end
    end


    imagesc(total_confusion_matrix)
    ax = gca;
    ax.XTickLabel = unique(testing_classes);
    ax.YTickLabel = unique(testing_classes);
    correct_predictions = sum(sum(total_confusion_matrix .* diag(ones(length(total_confusion_matrix),1)),1));
    total_predictions = sum(sum(total_confusion_matrix,1));
    correct_ratio = correct_predictions / total_predictions;
end
