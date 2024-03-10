% -----------------> train_ubms_timit.m <------------------- 

% The following code uses the MFCCs extracted from the train set of the 
% TIMIT dataset and uses them to train UBMs with the number of components
% being 1, 2, 4, 8 and 16.

% go to source directory
cd("..")

num_comp = [1, 2, 4, 8, 16];
filename = "data\UBM-TIMIT\MFCCs\timit_mfccs.mat";
timit_mfccs = importdata(filename);
% convert to num_framesx23 matrix
timit_mfccs = timit_mfccs';

for i = 1:length(num_comp)
    options = statset("Display", "off", "MaxIter", 500);
    ubm = fitgmdist(timit_mfccs, num_comp(i), "CovarianceType", "diagonal", "RegularizationValue", 1e-6, "Options", options);
        
    filename = "data\UBM-TIMIT\UBMs\ubm_" + num2str(num_comp(i)) + ".mat";
    save(filename, "ubm");
end

% go back to script folder
cd("gmm_training")