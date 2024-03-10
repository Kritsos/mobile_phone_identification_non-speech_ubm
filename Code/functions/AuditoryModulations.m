function [atm astm sqrate] = AuditoryModulations(Sig,fs,bts)


Sig = Sig - mean(Sig);

SongLength = length(Sig)/fs
NumOfBeats = length(bts);
BeatsIntv = SongLength/NumOfBeats

df = floor((BeatsIntv/7)*fs)

AS = LyonPassiveEar(Sig,fs,df); %3rd parameter 20

sqrate = size(AS,2)/SongLength
%% Modulation Scale Features
[row,col] = size(AS);

%%% based on continuous wavelet transform
for ind_row = 1:row
    envband_filter = AS(ind_row,:);
    %%%%%%%%%%%% CWT on the second transform %%%%%%%%%%%%%%%%%%%
    cwtFeature = isdlcwt(envband_filter,8);
    atm(ind_row,:,:) = cwtFeature;
end


 paras = [8     8    -2    -1]; %% Parameters
 rv = [2 4 8 16 32];
 sv = [0.25 0.5 1 2 4 8];  % scale vector (cycle/octave)
 astm = aud2cor(AS, paras, rv, sv, 'tmpxxx');
 
 atm = double(tenmat(atm,3))';
 astm = double(tenmat(abs(astm),4))';