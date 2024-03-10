% -----------------> SVM_classifier_audio_model_augmented2.m <------------------- 

% The following code uses the multiclass SVMs trained on the MFCCs of all
% the frames of the augmented2 dataset to classify each test speaker
% recording to one of the 21 phone models

% go to source directory
cd("..\..")

% number of components of the GMMs
num_comp = [1, 2, 4];
phones = {'HTC1', 'HTC2', 'LG1', 'LG2', 'LG3', 'LG4', 'N1', 'N2', 'N3', ...
          'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'SE1', 'SE2', ...
          'V1', 'iPH1'};

test_speakers = [1, 2, 3, 4, 7, 9, 12, 14, 17, 19, 22, 24];
% which subsets of the database to use
databases_names = {"baseline", "crop", "loudness", "pitch", "speed", "vtlp"};
% number of wav files per phone
samples = 12 * length(databases_names);
% non-speech / audio folder
f1 = "audio";
% which SVM folder
f2 = "augmented2";

% reshape to 1512x1 from 1x1512
y = reshape(repelem(1:21, samples), [], 1);
% label for each testing example that indicates from which subset the
% example comes (1: baseline, 2: crop, 3: loudness, 4: pitch, 5: speed, 6: vtlp)
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
    filename = "data\SVMs\" + f1 + "\" + f2 + "\" + num2str(num_comp(i)) + "\svm_model.mat";
    svm = importdata(filename);
    pred = predict(svm, x_mu);

    acc = sum(y == pred) / length(pred);
    acc_baseline = sum(y(subset == 1) == pred(subset == 1)) / 252;
    acc_crop = sum(y(subset == 2) == pred(subset == 2)) / 252;
    acc_loudness = sum(y(subset == 3) == pred(subset == 3)) / 252;
    acc_pitch = sum(y(subset == 4) == pred(subset == 4)) / 252;
    acc_speed = sum(y(subset == 5) == pred(subset == 5)) / 252;
    acc_vtlp = sum(y(subset == 6) == pred(subset == 6)) / 252;
    disp("Accuracy of " + num2str(num_comp(i)) + "-component means GSV SVM: " + num2str(100 * acc) + "%")
    disp("Baseline: " + num2str(100 * acc_baseline) + "% / " + "Crop: " + num2str(100 * acc_crop) + "% / " + "Loudness: " + ...
          num2str(100 * acc_loudness) + "% / " + "Pitch: " + num2str(100 * acc_pitch) + "% / " + "Speed: " + ...
          num2str(100 * acc_speed) + "% / " + "VTLP: " + num2str(100 * acc_vtlp) + "%")

    figure(2 * (i - 1) + 1)
    C = confusionmat(y, pred);
    confusionchart(C, phones);
    title(num2str("Model identification on audio frames with a " + num_comp(i)) + "-component means GSV SVM")

    % means and covariance
    filename = "data\SVMs\" + f1 + "\" + f2 + "\" + num2str(num_comp(i)) + "\svm_sigma_model.mat";
    svm = importdata(filename);
    pred = predict(svm, x_sigma);

    acc = sum(y == pred) / length(pred);
    acc_baseline = sum(y(subset == 1) == pred(subset == 1)) / 252;
    acc_crop = sum(y(subset == 2) == pred(subset == 2)) / 252;
    acc_loudness = sum(y(subset == 3) == pred(subset == 3)) / 252;
    acc_pitch = sum(y(subset == 4) == pred(subset == 4)) / 252;
    acc_speed = sum(y(subset == 5) == pred(subset == 5)) / 252;
    acc_vtlp = sum(y(subset == 6) == pred(subset == 6)) / 252;
    disp("Accuracy of " + num2str(num_comp(i)) + "-component means GSV SVM: " + num2str(100 * acc) + "%")
    disp("Baseline: " + num2str(100 * acc_baseline) + "% / " + "Crop: " + num2str(100 * acc_crop) + "% / " + "Loudness: " + ...
          num2str(100 * acc_loudness) + "% / " + "Pitch: " + num2str(100 * acc_pitch) + "% / " + "Speed: " + ...
          num2str(100 * acc_speed) + "% / " + "VTLP: " + num2str(100 * acc_vtlp) + "%")

    figure(2 * (i - 1) + 2)
    C = confusionmat(y, pred);
    confusionchart(C, phones);
    title(num2str("Model identification on audio frames with a " + num_comp(i)) + "-component means and covariance GSV SVM")
end

% return to script directory
cd("svm_classification\audio")