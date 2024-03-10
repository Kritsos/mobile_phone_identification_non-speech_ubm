function FeatureTensor = ComputeSTM(Signal,Fs,fcorname)

   
   Signal = resample(Signal,8000,Fs); %% Resampling
   Signal = unitseq(Signal);
   
   paras = [8     8    -2    -1]; %% Parameters
  
   rv = [2 4 8 16 32];
   sv = [0.25 0.5 1 2 4 8];  % scale vector (cycle/octave)
   ear = wav2aud(Signal, paras); 
   %aud_plot(ear, paras);
   ear = ear';
  
   cr = aud2cor(ear, paras, rv, sv, fcorname);
  
   rsf = mean(abs(cr), 4); %% rate, scale, frequency
   
   FeatureTensor = rsf;