% -----------------> train_ubms_timit2.m <------------------- 

% The following code uses the MFCCs extracted from the train set of the 
% TIMIT dataset and uses them to train a 64 component UBM. The MFCCs in
% used were extracted using the v_melcepst function of voicebox which
% include deltas and delta-deltas

% go to source directory
cd("..")

num_comp = [4, 64];
filename = "data2\UBM-TIMIT\MFCCs\timit_mfccs.mat";
timit_mfccs = importdata(filename);
% convert to num_framesx39 matrix
timit_mfccs = timit_mfccs';

for i = 1:length(num_comp)
    options = statset("Display", "off", "MaxIter", 500);
    ubm = fitgmdist(timit_mfccs, num_comp(i), "CovarianceType", "diagonal", "RegularizationValue", 1e-6, "Options", options);
            
    filename = "data2\UBM-TIMIT\UBMs\ubm_" + num2str(num_comp(i)) + ".mat";
    save(filename, "ubm");
end

% go back to script folder
cd("gmm_training")