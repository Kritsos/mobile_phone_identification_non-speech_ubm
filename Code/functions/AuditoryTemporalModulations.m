function [atm] = AuditoryTemporalModulations(Sig,fs)

AS = LyonPassiveEar(Sig,fs,20); %3rd parameter 20


%% Modulation Scale Features
[row,col] = size(AS);

%%% based on continuous wavelet transform
for ind_row = 1:row
    envband_filter = AS(ind_row,:);
    %%%%%%%%%%%% CWT on the second transform %%%%%%%%%%%%%%%%%%%
    cwtFeature = isdlcwt(envband_filter,8);
    atm(ind_row,:,:) = cwtFeature;
end


 