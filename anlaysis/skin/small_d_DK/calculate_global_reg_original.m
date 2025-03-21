function [imout,tform,cent,f,Rout]=calculate_global_reg_original(imrf,immv,rf,iternum,IHC)
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
    %figure(9),imshowpair(amv1out,arf)
    f=0;
%     figure(12),imshowpair(arf,amv1out),title(num2str(round(R*100)))
    % ct=0.92; % 0.8 CHANGE THIS IF NEED TO FLIP
    ct=0.75; % 0.8 CHANGE THIS IF NEED TO FLIP
    if R<ct
        disp('try flipping image')
        amv2=amv(end:-1:1,:,:);
        [R2,rs2,xy2,amv2out]=group_of_reg(amv2,arf,iternum0,sz,rf,bb);
        if R2>R;rs=rs2;xy=xy2;f=1;amv=amv2;disp(' chose flipped image');end
%         figure(13),imshowpair(arf,amv2out),title(num2str(round(R2*100)))
    end
    % rs=-90;
    % rs=70;
    % rs=20;
    % rs=30;
    % STOP HERE AND CHANGE RS
    [tform,amvout,~,~,Rout]=reg_ims_com(amv,arf,iternum-iternum0,sz,rf,rs,xy,0);
    figure,imshowpair(amvout(1:3:end,1:3:end,:),arf(1:3:end,1:3:end,:));
%     [tform,amvout,~,~,Rout]=reg_ims_com(amv,imrotate(arf,5),iternum-iternum0,sz,rf,rs,xy,0);
    %STOP HERE FOR TUNING OF ANGLE OF ROTATION
    aa=double(arf>0)+double(amvout>0);
    Rout=sum(aa(:)==2)/sum(aa(:)>0);
%     figure(9),imshowpair(amvout,arf)
    
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

