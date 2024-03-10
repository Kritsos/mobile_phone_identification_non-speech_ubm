% -----------------> SVM_classifier_non_speech_brand_pitch.m <------------------- 

% The following code uses the multiclass SVMs trained on the MFCCs of the
% non speech frames of the baseline + pitch dataset to classify each test 
% speaker recording to one of the 7 brands

% go to source directory
cd("..\..")

% number of components of the GMMs
num_comp = [1, 2, 4];
phones = {'HTC1', 'HTC2', 'LG1', 'LG2', 'LG3', 'LG4', 'N1', 'N2', 'N3', ...
          'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'SE1', 'SE2', ...
          'V1', 'iPH1'};
brands = {'HTC', 'LG', 'NOKIA', 'SAMSUNG', 'SONY_ERICSSON', 'VODAFONE', 'APPLE'};

test_speakers = [1, 2, 3, 4, 7, 9, 12, 14, 17, 19, 22, 24];
% which subsets of the database to use
databases_names = {"baseline", "pitch"};
% number of wav files per phone
samples = 12 * length(databases_names);
% non-speech / audio folder
f1 = "non-speech";
% which SVM folder
f2 = "pitch";

% reshape to 504x1 from 1x504
y = reshape([ones(1, 2 * samples), 2 * ones(1, 4 * samples), 3 * ones(1, 3 * samples), ...
                4 * ones(1, 8 * samples), 5 * ones(1, 2 * samples), 6 * ones(1, samples), ...
                7 * ones(1, samples)], [], 1);
% label for each testing example that indicates from which subset the
% example comes (1: baseline, 2: pitch)
subset = zeros(1, 21 * samples);

for i = 1:length(num_comp)
    % load test sets (only means and means and sigma)
    x_mu = zeros(21 * samples, 23 * num_comp(i));
    x_sigma = zeros(21 * samples, 2 * 23 * num_comp(i));
    x_counter = 1;

    for j = 1:length(phones)
        % folder of phone's num_comp(i)-component speaker GMMs
        folder = "data\GMMs\" + f1 + "\speakers\" + num2str(num_comp(i)) + "\" + phones{j} + "\";
    
        for k = 1:length(test_speakers)
            for m = 1:length(databases_names)
                filename = folder + "gmm_speaker_" + num2str(test_speakers(k)) + "_" + databases_names{m} + ".mat";
                gmm = importdata(filename);
        
                % reshape the means matrix from num_comp x 23 to a 23 *
                % num_comp vector containing the mean vectors stacked
                % horizontally
                x_mu(x_counter, :) = reshape(gmm.mu', 1, []);
                % also reshape the covariance matrix from 1x23xnum_comp to
                % 1x23*num_comp and horizontally stack it
                x_sigma(x_counter, :) = [reshape(gmm.mu', 1, []), reshape(gmm.Sigma, 1, [])];

                subset(x_counter) = m;
    
                x_counter = x_counter + 1;
            end
        end
    end
    
    % means only
    filename = "data\SVMs\" + f1 + "\" + f2 + "\" + num2str(num_comp(i)) + "\svm_brand.mat";
    svm = importdata(filename);
    pred = predict(svm, x_mu);

    acc = sum(y == pred) / length(pred);
    acc_baseline = sum(y(subset == 1) == pred(subset == 1)) / 252;
    acc_pitch = sum(y(subset == 2) == pred(subset == 2)) / 252;
    disp("Accuracy of " + num2str(num_comp(i)) + "-component means GSV SVM: " + num2str(100 * acc) + "%")
    disp("Baseline: " + num2str(100 * acc_baseline) + "% / " + "Pitch: " + num2str(100 * acc_pitch) + "%")

    figure(2 * (i - 1) + 1)
    C = confusionmat(y, pred);
    confusionchart(C, brands);
    title(num2str("Brand identification on non speech frames with a " + num_comp(i)) + "-component means GSV SVM")

    % means and covariance
    filename = "data\SVMs\" + f1 + "\" + f2 + "\" + num2str(num_comp(i)) + "\svm_sigma_brand.mat";
    svm = importdata(filename);
    pred = predict(svm, x_sigma);

    acc = sum(y == pred) / length(pred);
    acc_baseline = sum(y(subset == 1) == pred(subset == 1)) / 252;
    acc_pitch = sum(y(subset == 2) == pred(subset == 2)) / 252;
    disp("Accuracy of " + num2str(num_comp(i)) + "-component means and covariance GSV SVM: " + num2str(100 * acc) + "%")
    disp("Baseline: " + num2str(100 * acc_baseline) + "% / " + "Pitch: " + num2str(100 * acc_pitch) + "%")

    figure(2 * (i - 1) + 2)
    C = confusionmat(y, pred);
    confusionchart(C, brands);
    title(num2str("Brand identification on non speech frames with a " + num_comp(i)) + "-component means and covariance GSV SVM")
end

% return to script directory
cd("svm_classification\non-speech")