function cr = aud2cor(y, para1, rv, sv, fname, DISP)
% AUD2COR (forward) cortical rate-scale representation
%    cr = aud2cor(y, para1, rv, sv, fname, DISP);
%	cr	: cortical representation (4D, scale-rate(up-down)-time-freq.)
%	y	: auditory spectrogram, N-by-M, where
%		N = # of samples, M = # of channels
%	para1 = [paras FULLT FULLX BP]
%	 paras	: see WAV2AUD.
%	 FULLT (FULLX): fullness of temporal (spectral) margin. The value can
%		be any real number within [0, 1]. If only one number was
%		assigned, FULLT = FULLX will be set to the same value.
%	 BP	: pure bandpass indicator
%	 rv	: rate vector in Hz, e.g., 2.^(1:.5:5).
%	 sv	: scale vector in cyc/oct, e.g., 2.^(-2:.5:3).
%	 fname  : cortical representation file, if fname='tmpxxx', no data
%		  will be saved on disk to reduce processing time.
% 	 DISP	: saturation level for color panels. No display if 0.
%		If DISP < 0, every panel will be normalized by its max.
%
%	AUD2COR implements the 2-D wavelet transform
%	possibly executed by the A1 cortex. The auditory
%	spectrogram (Y) is the output generated by the 
%	cochlear model (WAV2AUD) according to the parameter
%	set PARA1. RV (SV) is the characteristic frequencies
%	(ripples) of the temporal (spatial) filters. This
%	function will store the output in a file with a
%	conventional extension .COR. Roughly, one-second
%	signal needs about 22 MB if 8-ms-frame is adopted.
%	Choosing truncated fashion (FULL = 0) will reduce
%	the size to 1/4, which will also reduce runing time
%	to half.
%	See also: WAV2AUD, COR_INFO, CORHEADR, COR_RST

% Auther: Powen Ru (powen@isr.umd.edu), NSL, UMD
% v1.00: 01-Jun-97
% v1.01: 11-Aug-97, function [corcplxw] 
% v1.02: 19-Aug-97, include FULL (0 < FULL < 1)
% v1.03: 28-Aug-97, include perfect reconstruction
% v1.04: 27-Sep-97, causal temporal filterbank; FULLT, FULLX
% v1.05: 11-Nov-97, bug in if length(para1) < 7
% v1.06: 12-Apr-98, removed non-causal option

% Revision: Taishih Chi (tschi@isr.umd.edu), NSL, UMD
% v1.10: 13-May-99, add cr, 4 dimensional representation

% Revision: Elena Grassi (egrassi@umd.edu), NSL, ISR, UMD
% v2.00: 23-Jun-04 faster implementation of 2D processing 
% Linearity of ifft is exploited to pull first inverse fft out of the loop
% for sdx = 1:K2 so the routine runs faster. 

if nargin < 6, DISP = 0; end;
if length(para1) < 5, FULLT = 0; else, FULLT = para1(5); end;
if length(para1) < 6, FULLX = FULLT; else, FULLX = para1(6); end;
if length(para1) < 7, BP = 0; else, BP = para1(7); end;

%FULLT, FULLX
% mean removal (for test only)
%meany   = mean(mean(y));
%y	   = y - meany;

% dimensions
K1 	= length(rv);	% # of rate channel
K2	= length(sv);	% # of scale channel
[N, M]	= size(y);	% dimensions of auditory spectrogram

% spatial, temporal zeros padding 
N1 = 2^nextpow2(N);	N2 = N1*2;
M1 = 2^nextpow2(M);	M2 = M1*2;

% first fourier transform (w.r.t. frequency axis)
Y = zeros(N2, M1);
for n = 1:N,
    R1 = fft(y(n, :), M2);
    Y(n, :) = R1(1:M1);
end;

% second fourier transform (w.r.t. temporal axis)
for m = 1:M1,
    R1 = fft(Y(1:N, m), N2);
    Y(:, m) = R1;
end;

paras = para1(1:4);		% parameters for aud. spectrogram
STF = 1000 / paras(1);	% frame per second
if (M == 96) SRF = 20;
else SRF = 24;		% channel per octave (fixed)
end

fout = fopen(fname, 'w');
fwrite(fout, [paras(:); K1; K2; rv(:); sv(:); N; M; FULLT; FULLX], ...
    'float');  

TMP = 0;        % default: write to HD
if length(fname)>=3
    TMP = strcmp(fname(1:3), 'tmp');
end;
% graphics
if DISP,
    load a1map_a;
    colormap(a1map);
end;

t0 = clock;

% freq. index
dM   = floor(M/2*FULLX);
mdx1 = [(1:dM)+M2-dM 1:M+dM];
mdx2 = [0 0 M+1 M+1 0]+dM;

% temp. index
dN   = floor(N/2*FULLT);
ndx  = 1:N+2*dN;
ndx1 = ndx;
ndx2 = [0 N+1 N+1 0 0];

z  = zeros(N+2*dN, M+2*dM);
cr = zeros(K2, K1*2, N+2*dN, M+2*dM);

for rdx = 1:K1,
    % rate filtering
    fc_rt = rv(rdx);
    HR = gen_cort(fc_rt, N1, STF, [rdx+BP K1+BP*2]); % if BP, all BPF 
    
    for sgn = [1 -1],
        
        % rate filtering modification
        if sgn > 0,
            HR = [HR; zeros(N1, 1)];	% SSB -> DSB
        else,
            HR = [HR(1); conj(flipud(HR(2:N2)))];
            %HR = [0; conj(flipud(HR(2:N2)))];
            HR(N1+1) = abs(HR(N1+2));
        end;
        
% % %         for sdx = 1:K2,
% % %             % scale filtering
% % %             fc_sc = sv(sdx);
% % %             
% % %             HS = gen_corf(fc_sc, M1, SRF, [sdx+BP K2+BP*2]);% if BP, all BPF
% % %             % spatiotemporal response
% % %             Z = (HR*HS') .* Y;
% % %             % first inverse fft (w.r.t. time axis)
% % %             for m = 1:M1,
% % %                 R1 = ifft(Z(:, m));
% % %                 z1(:, m) = R1(ndx1);
% % %             end;	% z1: N+2*dN -by- M1
% % %             
% % %             % second inverse fft (w.r.t frequency axis)
% % %             for n = ndx,
% % %                 R1 = ifft(z1(n, :), M2);
% % %                 z(n, :) = R1(mdx1);
% % %             end;	% z: N+2*dN -by- M+2*dM
% % %             % save file
% % %             cr(sdx, rdx+(sgn==1)*K1, :, :) = z;
% % %         end
        
        % The following code is equivalent to the commented out but more efficient since the first inverse fft is done out of the loop.
        
        % first inverse fft (w.r.t. time axis)
        z1= zeros(N2,M1); 
        for m = 1:M1, 
            z1(:,m)= HR.*Y(:,m);
        end;	
        z1= ifft(z1);
        z1= z1(ndx1,:);
        
        for sdx = 1:K2,
            % scale filtering
            fc_sc = sv(sdx);
            HS = gen_corf(fc_sc, M1, SRF, [sdx+BP K2+BP*2]);% if BP, all BPF
            
            % second inverse fft (w.r.t frequency axis)
            for n = ndx,
                R1 = ifft( (z1(n, :).*HS'), M2); 
                z(n, :) = R1(mdx1);
            end;	% z: N+2*dN -by- M+2*dM
            % save file
            cr(sdx, rdx+(sgn==1)*K1, :, :) = z;
            if ~TMP, 
                corcplxw(z, fout);
            end
            if DISP,
                image(cplx_col(z, DISP)');
                axis xy;
                
                if FULLT | FULLX, hold on;
                    plot(ndx2, mdx2, 'k--');
                    hold off; end;
                
                text('position', [N/2+dN, .9*M+dM*2], ...
                    'str', ['Scale = ' ...
                        sprintf('%5.2f', fc_sc) ...
                        ' cyc/oct, Rate = ' ...
                        sprintf('%5.2f', sgn*fc_rt) ...
                        ' Hz (Max. = ' ...
                        num2str(max(max(abs(z)))) ')'], ...
                    'ho', 'ce', 'fontwe', 'bold');
                drawnow;
            end;
        end;
    end;
    time_est(rdx, K1, 1, t0); 
    
end;

fclose(fout);

%%%%%%%%%%%%  End of AUD2COR %%%%%%%%%%%%%%
