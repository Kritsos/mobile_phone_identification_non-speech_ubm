
% 3D correlogram

[Signal,Fs,Nbits]= wavread('ClassicMono.wav');

Signal = Signal(1:44100);

ear = LyonPassiveEar(Signal, Fs,256);
figure;
sp = spectrogram1(Signal,256);

corEar = CorrelogramArray(ear,Fs,Fs/256,256);
corSp = CorrelogramArray(sp,Fs,Fs/256,256);

V = reshape(cor,256,16,58);

slice(V,1,1,1);