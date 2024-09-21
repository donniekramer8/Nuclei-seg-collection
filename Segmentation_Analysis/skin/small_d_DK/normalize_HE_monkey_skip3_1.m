function normalize_HE_monkey_skip3_1(pth,outpth)
warning ('off','all');
path(path,'\\motherserverdw\ashleysync\PanIN Modelling Package\'); 
path(path,'\\motherserverdw\ashleysync\PanIN Modelling Package\IHC segmentation\'); 
path(path,'\\fatherserverdw\andreex\docs\workflow codes\cell detection\')
% add in histogram equalization to make outputs better match

if ~exist('pthl','var');pthl=pth;end
if ~exist('outpth','var');outpth=[pth,'fix stain\'];end
outpthC=[outpth,'CVS\'];
outpthH=[outpth,'Hchannel\'];
outpthE=[outpth,'Echannel\'];
mkdir(outpth);mkdir(outpthC);mkdir(outpthH);mkdir(outpthE);


tic;
imlist=dir([pth,'*tif']);if size(imlist,1)<1;imlist=dir([pth,'*jpg']);end
knum=150000;

% H&E
CVS=[0.644 0.717 0.267;0.093 0.954 0.283;0.636 0.001 0.771];

% IHC bi-specific
% CVS=[0.650028 0.704031 0.286012 ;0.268147 0.570313 0.776427 ;0.711027 0.423181 0.561567];

% CVS=[0.578 0.738 0.348;...
%     0.410 0.851 0.328;...
%     0.588 0.625 0.514];
% CVS=[0.5157    0.7722    0.3712
%     0.3519    0.8409    0.4112
%     0.5662    0.1935    0.6088];
for kk=1:3:length(imlist)
        imnm=imlist(kk).name;
        if exist([outpthH,imnm],'file');disp(['skip ',num2str(kk)]);disp('skip');continue;end
        disp([num2str(kk), ' ', num2str(length(imlist)), ' ', imnm(1:end-4)]);
        
        im0=imread([pth,imnm]);
        %[CVS,TA]=make_rgb_kmeans_90(im0,knum,0);
        %CVS=[0.759 0.587 0.280;0.673 0.630 0.387;0.763 0.001 0.646]; % im1
        %save([outpthC,imnm(1:end-3),'mat'],'CVS');
        
        [imout,imH,imE]=colordeconv2pw4_log10(im0,"he",CVS);
        %load([outpthC,imlist(1).name(1:end-3),'mat'],'CVS');ODout=CVS;
        
%         figure(3),imshow(imH)
        imwrite(uint8(imH),[outpthH,imnm]);
        imwrite(uint8(imE),[outpthE,imnm]);
        disp([kk length(imlist)])
end
warning ('off','all');
end
