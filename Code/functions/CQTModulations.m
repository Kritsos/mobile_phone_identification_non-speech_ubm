function [cqttm cqtstm sqrate] = CQTModulations(Sig,fs,df,cf)

fmax = fs/2;     %center frequency of the highest frequency bin 
fmin = 57;
Sig = Sig - mean(Sig);

Xcqt = cqt(Sig,fmin,fmax,16,fs);
CQT = flipud(getCQT(Xcqt,'all','all'));

emptyHops = Xcqt.intParams.firstcenter/Xcqt.intParams.atomHOP;
maxDrop = emptyHops*2^(Xcqt.octaveNr-1)-emptyHops;
droppedSamples = (maxDrop-1)*Xcqt.intParams.atomHOP + Xcqt.intParams.firstcenter;
TimeVec = (1:size(CQT,2))*Xcqt.intParams.atomHOP-Xcqt.intParams.preZeros+droppedSamples;

CQT = CQT(:,1:df:end);
dfTimeVec = TimeVec(1:df:end);
CQT = CQT.^cf;

sqrate = size(CQT,2)/(max(dfTimeVec(:))/fs);
%% Modulation Scale Features
[row,col] = size(CQT);

%%% based on continuous wavelet transform
for ind_row = 1:row
    envband_filter = CQT(ind_row,:);
    %%%%%%%%%%%% CWT on the second transform %%%%%%%%%%%%%%%%%%%
    cwtFeature = isdlcwt(envband_filter,8);
    atm(ind_row,:,:) = cwtFeature;
end


 paras = [8     8    -2    -1]; %% Parameters
 rv = [2 4 8 16 32];
 sv = [0.25 0.5 1 2 4 8];  % scale vector (cycle/octave)
 astm = aud2cor(CQT, paras, rv, sv, 'tmpxxx');
 
 cqttm = double(tenmat(atm,3))';
 cqtstm = double(tenmat(abs(astm),4))';