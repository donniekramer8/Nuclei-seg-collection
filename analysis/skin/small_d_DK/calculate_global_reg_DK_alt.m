function [imout,tform_final,cent,Rout]=calculate_global_reg_DK_alt(imrf,immv,rf,iternum,IHC)

if IHC==1;bb=0.8;else;bb=0.9;end
% imrf == reference image
% immv == moving image
% szz == max size of images in stack
% rf == reduce images by _ times
% count == number of iterations of registration code
    % pre-registration image processing
    amv=imresize(immv,1/rf);amv=imgaussfilt(amv,2);
    arf=imresize(imrf,1/rf);arf=imgaussfilt(arf,2);
    sz=[0 0];cent=[0 0];
    if IHC>0;amv=imadjust(amv);arf=imadjust(arf);end
    
    % calculate registration, flipping image if necessary
    iternum0=2;
    [R,rs,xy,amv1out]=group_of_reg(amv,arf,iternum0,sz,rf,bb);

    [tform,amvout,~,~,Rout]=reg_ims_com(amv,arf,iternum-iternum0,sz,rf,rs,xy,0);
%     [tform,amvout,~,~,Rout]=reg_ims_com(amv,imrotate(arf,-60),iternum-iternum0,sz,rf,rs,xy,0);
    %STOP HERE FOR TUNING OF ANGLE OF ROTATION
    aa=double(arf>0)+double(amvout>0);
    Rout=sum(aa(:)==2)/sum(aa(:)>0);

    % figure(10),imshowpair(amvout,arf)
    
    % create output image
    Rin=imref2d(size(immv));
    if sum(abs(cent))==0
      mx=mean(Rin.XWorldLimits);
      my=mean(Rin.YWorldLimits);
      cent=[mx my];
    end

    % figure(9);
    % imshowpair(amvout, arf);
    % hold on;
    % plot(cent(1), cent(2), 'r+', 'MarkerSize', 10);
    % hold off;

    Rin.XWorldLimits = Rin.XWorldLimits-cent(1);
    Rin.YWorldLimits = Rin.YWorldLimits-cent(2);

    % tform_final = affine2d(tform.T * first_tform.T);
    
    % register
    % imout=imwarp(immv,Rin,tform_final,'nearest','Outputview',Rin,'Fillvalues',0);
    imout=imwarp(immv,Rin,tform,'nearest','Outputview',Rin,'Fillvalues',0);

    tform_final = tform;



    figure(9);
    imshow(imout);
    hold on;
    plot(cent(1), cent(2), 'r+', 'MarkerSize', 10);
    hold off;

    %figure,imshowpair(arf,amvout)
    % figure, imagesc(imout)
end
