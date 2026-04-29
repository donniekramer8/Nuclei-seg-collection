function [imout,tform_final,cent,Rout]=calculate_global_reg_DK_new(imrf,immv,rf,iternum,IHC)

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

    [flip_x, flip_y, max_angle] = find_best_angle(amv, arf);

    first_tform = affine2d();

    if flip_x
        amv = fliplr(amv);
        first_tform.T = [-1 0 0; 0 1 0; 0 0 1];
    elseif flip_y
        amv = flipud(amv);
        first_tform.T = [1 0 0; 0 -1 0; 0 0 1];
    end

    % if max_angle ~= 360 && max_angle ~= 0
    %     amv = imrotate(amv, max_angle, 'crop');
    %     theta = deg2rad(max_angle);
    %     rotationMatrix = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
    %     first_tform.T = first_tform.T*rotationMatrix;
    % end

    amv = imrotate(amv, max_angle, 'crop');
    theta = deg2rad(max_angle);
    rotationMatrix = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
    first_tform.T = first_tform.T*rotationMatrix;

    [y_rf, x_rf] = find(arf == 1);
    [y_mv, x_mv] = find(amv == 1);
    x_og = mean(x_mv) - mean(x_rf);
    y_og = mean(y_mv) - mean(y_rf);

    xy_og = [-x_og,-y_og];

    amv=imtranslate(amv,xy_og,'OutputView','same');

    
    % calculate registration, flipping image if necessary
    iternum0=2;
    [R,rs,xy,amv1out]=group_of_reg(amv,arf,iternum0,sz,rf,bb);

    xy = xy + xy_og';

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

    tform_final = affine2d(first_tform.T * tform.T);
    
    % register
    imout=imwarp(immv,Rin,tform_final,'nearest','Outputview',Rin,'Fillvalues',0);
    % imout=imwarp(immv,Rin,tform,'nearest','Outputview',Rin,'Fillvalues',0);

    % tform_final = tform;



    % figure(9);
    % imshow(imout);
    % hold on;
    % plot(cent(1), cent(2), 'r+', 'MarkerSize', 10);
    % hold off;

    %figure,imshowpair(arf,amvout)
    % figure, imagesc(imout)
end

