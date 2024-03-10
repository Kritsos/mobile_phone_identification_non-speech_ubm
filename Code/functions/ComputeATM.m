function [ATM] = ComputeATM(Signal,Fs,df,nbscales)
   
%% Auditory Spectrogram

ASG = LyonPassiveEar(Signal,Fs,df); %3rd parameter 20

%% Modulation Scale Features

[row,col] = size(ASG)

%%% based on continuous wavelet transform
for ind_row = 1:row
    envband_filter = ASG(ind_row,:);
    %%%%%%%%%%%% CWT on the second transform %%%%%%%%%%%%%%%%%%%
    cwtFeature = isdlcwt(envband_filter,nbscales);
    ATM(ind_row,:,:) = cwtFeature;
end

 
     

     
   
     

