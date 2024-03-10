% -----------------> SVM_classifier_ubm_adapted_brand2.m <------------------- 

% The following code uses the multiclass SVMs trained on the UBM means 
% adapted GSVs of the ccnu_mobile dataset to classify each test speaker
% recording to one of the 7 brands

% go to source directory
cd("..\..")

% number of components of the GMMs
num_comp = 4;
% the order of the phones must match the order of the brands further below
phones = {"AIR1", "AIR2_1", "AIR2_2", "iPAD7", "iPH6_1", "iPH6_2", "iPH6_3", "iPH6_4", "iPH6S_1", "iPH6S_2", "iPH6S_3", ...
          "iPH7P", "iPHSE", "iPHX", "H10", "H7X", "H8_1", "H8_2", "H8_3", "H9", "HV8", "HUAWN", "HUAWN2S", "HUAWN3E", ...
          "HUAWP10", "HUAWP20", "HUAWTAG", "NZ11", "OR9S", "S8", "SPH", "VX3F", "VX7", "VY11T", "MI2S", "MI5", "MI8", ...
          "MI8SE_1", "MI8SE_2", "MIX2", "R3S", "RNOTE3", "RNOTE4X", "ZC880A", "ZG719C"};
brands = {'APPLE', 'HONOR', 'HUAWEI', 'NUBIA', 'OPPO', 'SAMSUNG', 'VIVO', 'XIAOMI', 'ZTE'};

test_speakers = [3, 4, 5];
% number of wav files per phone
samples = 3;
% non-speech / audio folder
f = "audio";

% reshape to 135x1 from 1x135
y = reshape([ones(1, 14 * samples), 2 * ones(1, 7 * samples), 3 * ones(1, 6 * samples), ...
                4 * ones(1, samples), 5 * ones(1, samples), 6 * ones(1, 2 * samples), ...
                7 * ones(1, 3 * samples), 8 * ones(1, 9 * samples), 9 * ones(1, 2 * samples)], [], 1);

% load test sets
x_mu = zeros(45 * samples, 39 * num_comp);
x_counter = 1;

% load UBM
filename = "data2\UBM-TIMIT\UBMs\ubm_" + num2str(num_comp) + ".mat";
ubm = importdata(filename);

for j = 1:length(phones)
    for k = 1:length(test_speakers)
        % folder of phone's MFCCs
        filename = "data2\MFCCs\audio\" + phones{j} + "\" + "mfccs_speaker" + num2str(test_speakers(k)) + ".mat";
        mfccs = importdata(filename);

        % MAP adapt means of UBM to speaker's MFCCS
        mu = map_adapt_means(ubm, mfccs', 16);
        
        % reshape the means matrix from num_comp x 39 to a 39 *
        % num_comp vector containing the mean vectors stacked
        % horizontally
        x_mu(x_counter, :) = reshape(mu', 1, []);
    
         x_counter = x_counter + 1;
   end
end
    
% means only
filename = "data2\SVMs\ubm_adapted\svm_brand.mat";
svm = importdata(filename);
pred = predict(svm, x_mu);

acc = sum(y == pred) / length(pred);
disp("Accuracy of " + num2str(num_comp) + "-component means GSV SVM: " + num2str(100 * acc) + "%")
    
figure(1)
C = confusionmat(y, pred);
confusionchart(C, brands);
title(num2str("Brand identification on MAP adapted mean GSVs with a " + num_comp) + "-component UBM")

% return to script directory
cd("svm_classification\ubm_adapted2")