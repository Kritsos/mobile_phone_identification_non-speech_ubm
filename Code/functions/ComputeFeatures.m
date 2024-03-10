function [ATM] = ComputeFeatures(Signal, Fs)
   
%% Auditory Spectrogram
%Signal = Signal * (10^(96/20));
ear = LyonPassiveEar(Signal, Fs,20,8);

%% Modulation Scale Features
AuditoryModulationScalogram = modscale(ear);
ATM = AuditoryModulationScalogram;
 
     

     
   
     

