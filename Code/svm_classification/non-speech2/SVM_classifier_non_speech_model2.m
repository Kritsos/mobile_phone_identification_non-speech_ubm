% -----------------> SVM_classifier_non_speech_model2.m <------------------- 

% The following code uses the multiclass SVMs trained on the MFCCs of all
% the frames of the ccnu_mobile dataset to classify each test speaker
% recording to one of the 45 phone models

% go to source directory
cd("..\..")

% number of components of the GMMs
num_comp = 1;
phones = {'AIR1', 'AIR2_1', 'AIR2_2', 'iPAD7', 'iPH6_1', 'iPH6_2', 'iPH6_3', 'iPH6_4', 'iPH6S_1', 'iPH6S_2', 'iPH6S_3', ...
          'iPH7P', 'iPHSE', 'iPHX', 'H10', 'H7X', 'H8_1', 'H8_2', 'H8_3', 'H9', 'HV8', 'HUAWN', 'HUAWN2S', 'HUAWN3E', ...
          'HUAWP10', 'HUAWP20', 'HUAWTAG', 'NZ11', 'OR9S', 'S8', 'SPH', 'VX3F', 'VX7', 'VY11T', 'MI2S', 'MI5', 'MI8', ...
          'MI8SE_1', 'MI8SE_2', 'MIX2', 'R3S', 'RNOTE3', 'RNOTE4X', 'ZC880A', 'ZG719C'};
test_speakers = [3, 4, 5];
% number of wav files per phone
samples = 3;
% non-speech / audio folder
f = "non-speech";

% reshape to 135x1 from 1x135
y = reshape(repelem(1:45, samples), [], 1);

% load test sets (only means and means and sigma)
x_mu = zeros(45 * samples, 39 * num_comp);
x_sigma = zeros(45 * samples, 2 * 39 * num_comp);
x_counter = 1;

for j = 1:length(phones)
    % folder of phone's num_comp(i)-component speaker GMMs
    folder = "data2\GMMs\" + f + "\speakers\" + num2str(num_comp) + "\" + phones{j} + "\";
    
    for k = 1:length(test_speakers)
        filename = folder + "gmm_speaker_" + num2str(test_speakers(k)) + ".mat";
        gmm = importdata(filename);
        
        % reshape the means matrix from num_comp x 39 to a 39 *
        % num_comp vector containing the mean vectors stacked
        % horizontally
        x_mu(x_counter, :) = reshape(gmm.mu', 1, []);
        % also reshape the covariance matrix from 1x39xnum_comp to
        % 1x39*num_comp and horizontally stack it
        x_sigma(x_counter, :) = [reshape(gmm.mu', 1, []), reshape(gmm.Sigma, 1, [])];

        x_counter = x_counter + 1;
    end
end
    
% means only
filename = "data2\SVMs\" + f + "\svm_model.mat";
svm = importdata(filename);
pred = predict(svm, x_mu);

acc = sum(y == pred) / length(pred);
disp("Accuracy of " + num2str(num_comp) + "-component means GSV SVM: " + num2str(100 * acc) + "%")

figure(1)
C = confusionmat(y, pred);
confusionchart(C, phones);
title(num2str("Model identification on audio frames with a " + num_comp) + "-component means GSV SVM")

% means and covariance
filename = "data2\SVMs\" + f + "\svm_sigma_model.mat";
svm = importdata(filename);
pred = predict(svm, x_sigma);

acc = sum(y == pred) / length(pred);
disp("Accuracy of " + num2str(num_comp) + "-component means and covariance GSV SVM: " + num2str(100 * acc) + "%")

figure(2)
C = confusionmat(y, pred);
confusionchart(C, phones);
title(num2str("Model identification on audio frames with a " + num_comp) + "-component means and covariance GSV SVM")

% return to script directory
cd("svm_classification\non-speech2")