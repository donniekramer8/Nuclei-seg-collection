function [TA]=find_tissue_area_organoids(im0)
if ~exist('IHC','var');IHC=0;end

    %im=double(im0);

    im=double(im0);
    
    im1xg=imgaussfilt(double(im),2);
    
    % get dark objects
    ima=im(:,:,2);
    TA=ima<230;
    
    % remove black objects
    imb=std(im1xg,[],3);
    imc=abs(double(im(:,:,1))-double(im(:,:,2)));
    TA=TA & imc>7;
    TA=imclose(TA,strel('disk',4));
    TA=bwareaopen(TA,500);
    TA=imclose(TA,strel('disk',10));
    TA=imdilate(TA,strel('disk',3));
    TA=imfill(TA,'holes');
    

end
