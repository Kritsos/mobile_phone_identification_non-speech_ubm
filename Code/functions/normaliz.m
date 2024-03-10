function [ndat,norms] = normaliz(dat);
%USAGE:
%        [ndat,norms] = normaliz(dat);
%
%DSCRIPTION: Normalizes rows of matrix to unit vectors.
%           .
%
%__________________________________________________________________________
%
%INPUT:      dat: Data matrix .
%                       
%__________________________________________________________________________
%
%OUTPUT:      ndat: Matrix of normalized data.
%            norms: The vector of norms used in the normalization.
%__________________________________________________________________________


% ________________/Author\___________________
% Yannis Panagakis, Ph.D. student, [AIIA Lab]
% Dept. Informatics, AUTH - GR.
% email: yannisp@csd.auth.gr
%____________________________________________

[m,n] = size(dat);
ndat = dat;
norms = zeros(m,1);
for i = 1:m
  if norm(ndat(i,:)) ~= 0
    norms(i) = norm(ndat(i,:));
    ndat(i,:) = ndat(i,:)/norms(i);
  else
    disp(sprintf('The norm of sample %g is 0, sample not normalized!',i))
  end
end
