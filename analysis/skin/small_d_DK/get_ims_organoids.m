function [im,TA]=get_ims_organoids(pth,nm,tp,IHC)
if ~exist([pth,'TA\'],'dir');mkdir([pth,'TA\']);end


im=imread([pth,nm,tp]);
if size(im,3)==1;im=cat(3,im,im,im);end
pthTA=[pth,'TA\'];
if exist([pthTA,nm,'tif'],'file')
    TA=imread([pthTA,nm,'tif']);
else
%     TA=find_tissue_area(im,IHC);
    TA=find_tissue_area_organoids(im);
    imwrite(TA,[pthTA,nm,'tif']);
end
TA=uint8(TA>0);
