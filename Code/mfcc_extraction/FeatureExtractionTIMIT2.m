function FeatureExtractionTIMIT2()
%Batch processing: Returns the mat file of all mffccs to be used for 
% training a UBM on TIMIT
% FeatureExtractionTIMIT;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Uses the voicebox v_melcepst function to extract the MFCCs (and deltas 
% and deltas-deltas) and performs mean and variance normalization on them
allmfccs=cell(6300, 1);
mfccs_counter = 1;
Fs=16000;

cd('../TIMIT/TRAIN');
Folders = dir;
NumOfFolders = length(Folders)-2; %should be 8
disp(['NumOfFolder= ' num2str(NumOfFolders)]);
for i = 1:NumOfFolders
        cd([Folders(i+2).name]) % move to each DR
        disp([Folders(i+2).name]);
        SubFolders = dir;
        NumOfSubFolders = length(SubFolders)-2;
        for ii = 1:NumOfSubFolders 
            cd([SubFolders(ii+2).name]) % move to each speaker directory
            disp([SubFolders(ii+2).name]);
            FileList = dir('*.WAV');
            NumOfFiles = length(FileList) - 2;
            for iii = 1:NumOfFiles % for each utterance                       
                %% Read the audio file.
                FileName = FileList(iii+2).name;
                %disp([FileName]);
                Sig=audioread(FileName);
                %% Compute MFCCs
                mfccs = v_melcepst(Sig, Fs, '0dD');
                mfccs = mfccs';
                % cepstral mean and variance normalization
                % mfccs = cmvn(mfccs', true);
                allmfccs{mfccs_counter} = mfccs;
                mfccs_counter = mfccs_counter + 1;
            end
            cd ../
        end
        cd ../ 
end
% return back to source folder
cd ../../
% convert to 39xtotal number of frames matrix
allmfccs = [allmfccs{:}];
filename = "data2\UBM-TIMIT\MFCCs\timit_mfccs.mat";
save(filename,"allmfccs");

% return to mfcc_extraction folder
cd("mfcc_extraction")
end