% -----------------> SVM_classifier_ubm_adapted_model_baseline.m <------------------- 

% The following code uses the multiclass SVMs trained on the UBM means 
% adapted GSVs of the baseline dataset to classify each test speaker
% recording to one of the 21 models

% go to source directory
cd("..\..")

% number of components of the GSVs
num_comp = [1, 2, 4, 8, 16];
phones = {'HTC1', 'HTC2', 'LG1', 'LG2', 'LG3', 'LG4', 'N1', 'N2', 'N3', ...
          'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'SE1', 'SE2', ...
          'V1', 'iPH1'};

test_speakers = [1, 2, 3, 4, 7, 9, 12, 14, 17, 19, 22, 24];
% which subsets of the database to use
databases_names = {"baseline"};
% number of wav files per phone
samples = 12 * length(databases_names);
% which SVM folder
f = "baseline";

% reshape to 252x1 from 1x252
y = reshape(repelem(1:21, samples), [], 1);

for i = 1:length(num_comp)
    % load test sets
    x_mu = zeros(21 * samples, 23 * num_comp(i));
    x_counter = 1;

    % load UBM
    filename = "data\UBM-TIMIT\UBMs\ubm_" + num2str(num_comp(i)) + ".mat";
    ubm = importdata(filename);

    for j = 1:length(phones)
        for k = 1:length(test_speakers)
            for m = 1:length(databases_names)
                % folder of phone's MFCCs
                filename = "data\MFCCs\audio\" + databases_names{m} + "\test\" + phones{j} + "\" + "mfccs_speaker" + num2str(test_speakers(k))...
                            + ".mat";
                mfccs = importdata(filename);

                % MAP adapt means of UBM to speaker's MFCCS
                mu = map_adapt_means(ubm, mfccs', 16);
        
                % reshape the means matrix from num_comp x 23 to a 23 *
                % num_comp vector containing the mean vectors stacked
                % horizontally
                x_mu(x_counter, :) = reshape(mu', 1, []);
    
                x_counter = x_counter + 1;
            end
        end
    end
    
    % means only
    filename = "data\SVMs\ubm_adapted\" + f + "\" + num2str(num_comp(i)) + "\svm_model.mat";
    svm = importdata(filename);
    pred = predict(svm, x_mu);

    acc = sum(y == pred) / length(pred);
    disp("Accuracy of " + num2str(num_comp(i)) + "-component means GSV SVM: " + num2str(100 * acc) + "%")

    figure(i)
    C = confusionmat(y, pred);
    confusionchart(C, phones);
    title(num2str("Model identification on MAP adapted mean GSVs with a " + num_comp(i)) + "-component UBM")
end

% return to script directory
cd("svm_classification\ubm_adapted")