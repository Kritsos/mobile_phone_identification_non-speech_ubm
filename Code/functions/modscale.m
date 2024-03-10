function ModulationScale  = modscale(ear,nbscales)

S = ear;
[row,col] = size(S);



%%% based on DWT
   [row,col] = size(S);
   for ind_row = 1:row
       envband_filter = S(ind_row,:);
       %%%%%%%%%%%% CWT on the second transform %%%%%%%%%%%%%%%%%%%
       envband_filter = envband_filter-mean(envband_filter); 
       [C,L] = wavedec(envband_filter,nbscales,'bior5.5');           %%matlab command
       L = L(1:end-1);         %% cut the last one which is the length of x
       C = fliplr(C);
       L = fliplr(L);
       start = 1;
       stop = L(1);
       dwtfeature = [];
       for ind = 1:length(L)
           Cscale = C(start:stop);
           dwtfeature(ind) = sum(Cscale.^2);
           start = stop+1;
           if (ind < length(L))
               stop = stop+L(ind+1);
           else
               stop = length(C);
           end
       end
       ModscaleDWT(ind_row,:,:) = dwtfeature;
    end
    %%%% normally I don't use the low pass feature (stationary features)
    ModulationScale = ModscaleDWT;%ModscaleDWT(:,1:end-1);

  

     


