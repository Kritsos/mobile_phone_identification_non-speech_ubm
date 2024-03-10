% -----------------> train_gmms_non_speech_speakers.m <------------------- 

% The following code uses the extracted training and testing non-speech MFCCs
% of all the subsets of the dataset and uses them to train speaker dependent
% GNMs (for the train and the test set) which will be used for GSV
% extraction in the SVM training

% number of components of the GMMs
num_comp = [1, 2, 4, 8, 16, 32, 64];
phones = {"HTC1", "HTC2", "LG1", "LG2", "LG3", "LG4", "N1", "N2", "N3", ...
          "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "SE1", "SE2", ...
          "V1", "iPH1"};

train_speakers = [5, 6, 8, 10, 11, 13, 15, 16, 18, 20, 21, 23];
test_speakers = [1, 2, 3, 4, 7, 9, 12, 14, 17, 19, 22, 24];
databases_names = {"baseline", "gaussian", "background", "reverberation", "crop", "loudness", "pitch", "speed", "vtlp"};

% go to source directory
cd("..\..")

% 1: train, 2: test
for n = 1:2
    if(n == 1)
        f = "train\";
        speakers = train_speakers;
    else
        f = "test\";
        speakers = test_speakers;
    end

    % for each database
    for m = 1:length(databases_names)
        % for each phone
        for i = 1:length(phones)
            % get folder of testing audio of phone in current subset
            folder = "data\MFCCs\non-speech\" + databases_names{m} + "\" + f + phones{i} + "\";
        
            % read the MFCCs of each speaker
            for j = 1:length(speakers)
                filename = folder + "mfccs_speaker" + num2str(speakers(j)) + ".mat";
                mfccs = importdata(filename);
                % convert to numframesx23 matrix
                mfccs = mfccs';
                
                % minimum mfccs frames number encountered was 32 (in the 
                % cropped subset) so we duplicate the mfcc matrix for the 
                % 64-component GMM
                if(length(mfccs) < 64)
                    mfccs = [mfccs;mfccs;mfccs(1, :)];
                end
                % train the GMMs with the different number of components
                for k = 1:length(num_comp)
                    options = statset("Display", "off", "MaxIter", 500);
                    gmm = fitgmdist(mfccs, num_comp(k), "CovarianceType", "diagonal", "RegularizationValue", 1e-6, "Options", options);
                    
                    filename = "data\GMMs\non-speech\speakers\" + num2str(num_comp(k)) + "\" + phones{i} + "\gmm_speaker_" + num2str(speakers(j)) ...
                                + "_" + databases_names{m} + ".mat";
                    save(filename, "gmm");
                end
            end
        end
    end
end

% return to script directory
cd("gmm_training\non-speech")