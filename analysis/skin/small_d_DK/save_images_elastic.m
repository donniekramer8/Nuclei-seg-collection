function save_images_elastic(pthim,pthdata,scale,padnum,redo,cropim)
if ~exist('redo','var');redo=0;end
if ~exist('padnum','var');pd=1;padnum=[];else;pd=0;end
if isempty(padnum);pd=1;end
if ~exist('cropim','var');cropim=0;end
imlist=dir([pthim,'*tif']);fl='tif';
if isempty(imlist);imlist=dir([pthim,'*jp2']);fl='jp2';end
if isempty(imlist);imlist=dir([pthim,'*jpg']);fl='jpg';end

matlist=dir([pthdata,'D\','*mat']);
try 
    datafileE=[pthdata,matlist(1).name];
    load(datafileE,'szz','padall');
catch
    datafileE=[pthdata,matlist(end).name];
    load(datafileE,'szz','padall');
end
padall=ceil(padall*scale);
refsize=ceil(szz*scale);

% determine crop region
if cropim~=0
    if length(cropim)==1
        [rot,rr]=get_cropim(pthdata,scale);
    else
        rot=cropim(1);rr=cropim(2:end);
    end
end

% determine roi and create output folder
outpth=[pthim,'registeredE\'];
if ~isfolder(outpth);mkdir(outpth);end

% register each image and save to outpth
for kz=1:length(matlist)
    imnm=[matlist(kz).name(1:end-3),fl];
    outnm=imnm;
    if exist([outpth,outnm],'file') && ~redo;disp(['skip image ',num2str(kz)]);continue;end
    %if exist([outpth,outnm],'file');disp('calculating again');end
    %if contains(imnm,'CD');continue;end
    if ~exist([pthim,imnm],'file');continue;end
    datafileE=[pthdata,imnm(1:end-3),'mat'];
    datafileD=[[pthdata,'D\'],imnm(1:end-3),'mat'];
    if ~exist(datafileD,'file');continue;end
    f=0;
    
    % load image
    IM=imread([pthim,imnm]);
    szim=size(IM(:,:,1));
    if pd;padnum=squeeze(mode(mode(IM,2),1))';end
    if szim(1)>refsize(1) || szim(2)>refsize(2)
        a=min([szim; refsize]);
        IM=IM(1:a(1),1:a(2),:);
    end
    IM=pad_im_both2(IM,refsize,padall,padnum);
    
    
    % if not reference image, register
    try
        load(datafileE,'tform','cent','f');
        if f==1;IM=IM(end:-1:1,:,:);end
        IMG=register_IM(IM,tform,scale,cent,padnum);

        load(datafileD,'D');
        D2=imresize(D,size(IM(:,:,1)));
        D2=D2.*scale;
        IME=imwarp(IMG,D2,'nearest','FillValues',padnum);
    % no transformation if this is reference image
    catch
        IME=IM;
    end
    
    if kz==1
       pth1=[pthdata,'..\'];
       try 
           im=imread([pth1,matlist(kz).name(1:end-3),'jpg']);
       catch
           im=imread([pth1,matlist(kz).name(1:end-3),'tif']);
       end
       im2=imresize(IME,size(im(:,:,1)),'nearest');
       figure,imshowpair(im,im2);pause(2);
    end
    
    if cropim
        IME=imrotate(IME,rot,'nearest');
        IME=imcrop(IME,rr);
        if kz==1;save([outpth,'crop_data.mat'],'rot','rr');end
    end
    
    imwrite(IME,[outpth,outnm]);
    disp([kz length(imlist)]);%disp(imnm)
    clearvars tform rsc cent D
end

end

function IM=register_IM(IM,tform,scale,cent,abc)    
    % rough registration
    cent=cent*scale;
    tform.T(3,1:2)=tform.T(3,1:2)*scale;
    Rin=imref2d(size(IM));
        Rin.XWorldLimits = Rin.XWorldLimits-cent(1);
        Rin.YWorldLimits = Rin.YWorldLimits-cent(2);
    IM=imwarp(IM,Rin,tform,'nearest','outputview',Rin,'fillvalues',abc);
end


function [rot,rr]=get_cropim(pthdata,scale)

    pth1=[pthdata,'..\'];
    imlist=dir([pth1,'*tif']);if isempty(imlist);imlist=dir([pth1,'*jpg']);end
    im1=rgb2gray(imread([pth1,imlist(1).name]));
    im2=rgb2gray(imread([pth1,imlist(round(length(imlist)/2)).name]));
    im3=rgb2gray(imread([pth1,imlist(end).name]));
    im=cat(3,im1,im2,im3);
    h=figure;imshow(im);isgood=0;
    while isgood~=1
        rot=input('angle?\n');
        imshow(imrotate(im,rot));
        isgood=input('is good?\n');
    end
    im=imrotate(im,rot,'nearest');
    [~,rr]=imcrop(im);
    rr=round(rr)*scale;
    close(h)

end


