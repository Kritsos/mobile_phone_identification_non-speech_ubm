function [coefs] = isdlcwt(signal,nbscales)

signal = signal(:)';
len    = length(signal);
%nbscales = 8;
%load coefCWT_scale128;
load coefcwt_bior55;
sig_dm = signal-mean(signal);
scale_start = 1;

fftlen = 2^ceil(log2(length(signal)+Lpsi(nbscales)-1));
X = fft([sig_dm, zeros(1, fftlen-length(sig_dm))]);
for k = scale_start:nbscales
    f = psivals(k,1:Lpsi(k));
%    f = f./sqrt(sum(f.^2)); % Normalize basis functions to have unit energy
    Y = fft([f, zeros(1, fftlen-length(f))]);
    fsig = real(ifft(X.*Y));
    fsig = fsig(1:length(signal)+length(f)-1);
    cwtcoeff = wkeep(diff(fsig),len);
    coefs(k-scale_start+1,:) = cwtcoeff.^2;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find lowpass
%f = phivals;
%cwtcoeff = wkeep(diff(conv(signal,f)),len);
%coefs(k+1) = sum(cwtcoeff.^2);
%coefs(k-scale_start+2) = sum(signal);
