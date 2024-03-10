% -----------------> extract_mfccs_non_speech.m <------------------- 

% The following code extracts the MFCCs of the silent frames of the train
% and test speakers for each subset of the augmented dataset. For each phone 
% and for each speaker (for the train and the test set) it saves a 23xnumframes 
% matrix containing the corresponding MFCCs

% 6 male and 6 female speakers chosen to be the training set for each cell phone
train_speakers = {"speaker5.wav";"speaker6.wav";"speaker8.wav";"speaker10.wav";...
"speaker11.wav";"speaker13.wav";"speaker15.wav";"speaker16.wav";"speaker18.wav";...
"speaker20.wav";"speaker21.wav";"speaker23.wav"};

% remaining 6 male and 6 female speakers are the test set for each cell phone
test_speakers = {"speaker1.wav";"speaker2.wav";"speaker3.wav";"speaker4.wav";...
"speaker7.wav";"speaker9.wav";"speaker12.wav";"speaker14.wav";"speaker17.wav";...
"speaker19.wav";"speaker22.wav";"speaker24.wav"};

% databases location
databases = {"MOBIPHONE\", "MOBIPHONE_GAUSSIAN\", "MOBIPHONE_BACKGROUND\", "MOBIPHONE_REVERBERATION\", ...
             "MOBIPHONE_CROP\", "MOBIPHONE_LOUDNESS\", "MOBIPHONE_PITCH\", "MOBIPHONE_SPEED\", ...
             "MOBIPHONE_VTLP\"};
databases_names = {"baseline", "gaussian", "background", "reverberation", "crop", "loudness", "pitch", "speed", "vtlp"};

% the dir() function sorts the files and folders alphabetically with
% capital letters being first so this is the order in which the phones are
% read: HTC desire c, HTC sensation xe, LG GS290, LG L3, LG Optimus L5,
% LG Optimus L9, Nokia 5530, Nokia C5, Nokia N70, Samsung E2121B,
% Samsung E2600, Samsung GT-I8190 mini, Samsung GT-N7100 (galaxy note2),
% Samsung Galaxy GT-I9100 s2, Samsung Galaxy Nexus S, Samsung e1230,
% Samsung s5830i, Sony Ericson c902, Sony ericson c510i, Vodafone joy 845,
% iPhone5 and so the below IDs correspond to these phones in the above
% order
phones = {"HTC1", "HTC2", "LG1", "LG2", "LG3", "LG4", "N1", "N2", "N3", ...
          "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "SE1", "SE2", ...
          "V1", "iPH1"};

% go to source directory
cd("..")

for m = 1:length(databases)
    d = dir(databases{m});

    % 1 = train, 2 = test
    for n = 1:2
        if(n == 1)
            folder = "train\";
            speakers = train_speakers;
        else
            folder = "test\";
            speakers = test_speakers;
        end
    
        for i = 3:numel(d)
            % path of phone's audio files
            path = databases{m} + d(i).name + "\";
            cd(path)
            
            % for each speaker
            for j = 1:12
                % read wav file
                [x, fs] = audioread(speakers{j});         
                 
                 % speech probability threshold
                 if(m == 2 || m == 8) % gaussian or speed
                    threshold = 0.5;
                 else
                     if(m == 3) % background
                        threshold = 0.8;
                     else % rest
                         threshold = 0.7;
                     end
                 end

                 % Perform VAD: default parameters perform VAD on 10ms non
                 % overlapping frames
                 sp = v_vadsohn(x(:, 1), fs, 'b'); % speech activity likelihood ratio
            
                 % extract non speech parts of the signal
                 x = x(sp < threshold, 1);
                 % extract MFCCs on 20ms frames with 10ms hop on the estimated non speech parts
                 y = extract_features(x, fs);
        
                 % go back to initial directory to save the MFCCs
                 cd("..\..\") 
                 filename = "data\MFCCs\non-speech\" + databases_names{m} + "\" + folder + phones{i - 2} + "\mfccs_" + speakers{j}{1}(1:end-4) + ".mat";
                 save(filename, "y");
                 % go into the folder with the phone's audio files again
                 cd(path)
            end
            % go back to initial directory
            cd("..\..\")
        end
    end
end

% return to script directory
cd("mfcc_extraction")