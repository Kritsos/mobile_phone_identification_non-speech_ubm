% -----------------> train_svms_ubm_adapted2.m <------------------- 

% The following code uses the UBM adapted speaker GMMs trained on the MFCCs 
% of all the frames of the ccnu_mobile dataset to extract the training GSVs 
% and trains brand and model SVMs using bayesian optimization for hyperparameter 
% tuning. The means of the GMMs are only adapted so we only use mean GSVs

% go to source directory
cd("..\..")

% number of components of the GSVs
num_comp = 4;
% the order of the phones must match the order of the brands further below
phones = {"AIR1", "AIR2_1", "AIR2_2", "iPAD7", "iPH6_1", "iPH6_2", "iPH6_3", "iPH6_4", "iPH6S_1", "iPH6S_2", "iPH6S_3", ...
          "iPH7P", "iPHSE", "iPHX", "H10", "H7X", "H8_1", "H8_2", "H8_3", "H9", "HV8", "HUAWN", "HUAWN2S", "HUAWN3E", ...
          "HUAWP10", "HUAWP20", "HUAWTAG", "NZ11", "OR9S", "S8", "SPH", "VX3F", "VX7", "VY11T", "MI2S", "MI5", "MI8", ...
          "MI8SE_1", "MI8SE_2", "MIX2", "R3S", "RNOTE3", "RNOTE4X", "ZC880A", "ZG719C"};

train_speakers = [1, 2];
% number of wav files per phone
samples = 2;
% non-speech / audio folder
f = "audio";

y_model = repelem(1:45, samples);
% 1: APPLE, 2: HONOR, 3: HUAWEI, 4: NUBIA, 5: OPPO, 6: SAMSUNG, 7: VIVO, 8: XIAOMI, 9: ZTE
y_brand = [ones(1, 14 * samples), 2 * ones(1, 7 * samples), 3 * ones(1, 6 * samples), ...
                4 * ones(1, samples), 5 * ones(1, samples), 6 * ones(1, 2 * samples), ...
                7 * ones(1, 3 * samples), 8 * ones(1, 9 * samples), 9 * ones(1, 2 * samples)];


% load train sets
x_mu = zeros(45 * samples, 39 * num_comp);
x_counter = 1;

% load UBM
filename = "data2\UBM-TIMIT\UBMs\ubm_" + num2str(num_comp) + ".mat";
ubm = importdata(filename);

for j = 1:length(phones)
    for k = 1:length(train_speakers)
        % folder of phone's MFCCs
        filename = "data2\MFCCs\audio\" + phones{j} + "\" + "mfccs_speaker" + num2str(train_speakers(k)) + ".mat";
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
    
% train brand and model SVMs using bayesian optimization for
% hyperparameter tuning

svm_brand = fitcecoc(x_mu, y_brand, 'Learners', 'svm', 'OptimizeHyperparameters', 'all', ...
                    'HyperparameterOptimizationOptions',struct('ShowPlots', false, 'UseParallel', true));
filename = "data2\SVMs\ubm_adapted\svm_brand.mat";
save(filename, "svm_brand")

svm_model = fitcecoc(x_mu, y_model, 'Learners', 'svm', 'OptimizeHyperparameters', 'all', ...
                    'HyperparameterOptimizationOptions',struct('ShowPlots', false, 'UseParallel', true));
filename = "data2\SVMs\ubm_adapted\svm_model.mat";
save(filename, "svm_model")

% return to script directory
cd("svm_training\ubm_adapted2")