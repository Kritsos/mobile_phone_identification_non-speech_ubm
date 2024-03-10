% -----------------> SVM_classifier_ubm_adapted_brand_gaussian.m <------------------- 

% The following code uses the multiclass SVMs trained on the UBM means 
% adapted GSVs of the baseline + gaussian dataset to classify each test 
% speaker recording to one of the 7 brands

% go to source directory
cd("..\..")

% number of components of the GSVs
num_comp = [1, 2, 4, 8, 16];
phones = {'HTC1', 'HTC2', 'LG1', 'LG2', 'LG3', 'LG4', 'N1', 'N2', 'N3', ...
          'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'SE1', 'SE2', ...
          'V1', 'iPH1'};
brands = {'HTC', 'LG', 'NOKIA', 'SAMSUNG', 'SONY_ERICSSON', 'VODAFONE', 'APPLE'};

test_speakers = [1, 2, 3, 4, 7, 9, 12, 14, 17, 19, 22, 24];
% which subsets of the database to use
databases_names = {"baseline", "gaussian"};
% number of wav files per phone
samples = 12 * length(databases_names);
% which SVM folder
f = "gaussian";

% reshape to 504x1 from 1x504
y = reshape([ones(1, 2 * samples), 2 * ones(1, 4 * samples), 3 * ones(1, 3 * samples), ...
                4 * ones(1, 8 * samples), 5 * ones(1, 2 * samples), 6 * ones(1, samples), ...
                7 * ones(1, samples)], [], 1);
% label for each testing example that indicates from which subset the
% example comes (1: baseline, 2: gaussian)
subset = zeros(1, 21 * samples);

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

                subset(x_counter) = m;
    
                x_counter = x_counter + 1;
            end
        end
    end
    
    % means only
    filename = "data\SVMs\ubm_adapted\" + f + "\" + num2str(num_comp(i)) + "\svm_brand.mat";
    svm = importdata(filename);
    pred = predict(svm, x_mu);

    acc = sum(y == pred) / length(pred);
    acc_baseline = sum(y(subset == 1) == pred(subset == 1)) / 252;
    acc_gaussian = sum(y(subset == 2) == pred(subset == 2)) / 252;
    disp("Accuracy of " + num2str(num_comp(i)) + "-component means GSV SVM: " + num2str(100 * acc) + "%")
    disp("Baseline: " + num2str(100 * acc_baseline) + "% / " + "Gaussian: " + num2str(100 * acc_gaussian) + "%")
    
    figure(i)
    C = confusionmat(y, pred);
    confusionchart(C, brands);
    title(num2str("Brand identification on MAP adapted mean GSVs with a " + num_comp(i)) + "-component UBM")
end

% return to script directory
cd("svm_classification\ubm_adapted")