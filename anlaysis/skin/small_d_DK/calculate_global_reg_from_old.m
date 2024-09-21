function [imout,tform,cent,f,Rout]=calculate_global_reg_from_old(imrf,immv,rf,iternum,IHC) 
    % create output image
    Rin=imref2d(size(immv));
    if sum(abs(cent))==0
      mx=mean(Rin.XWorldLimits);
      my=mean(Rin.YWorldLimits);
      cent=[mx my];
    end
    Rin.XWorldLimits = Rin.XWorldLimits-cent(1);
    Rin.YWorldLimits = Rin.YWorldLimits-cent(2);

    if f==1
        immv=immv(end:-1:1,:,:);
    end
    
    % register
    imout=imwarp(immv,Rin,tform,'nearest','Outputview',Rin,'Fillvalues',0);
    %figure,imshowpair(arf,amvout)
    % figure, imagesc(imout)
end
