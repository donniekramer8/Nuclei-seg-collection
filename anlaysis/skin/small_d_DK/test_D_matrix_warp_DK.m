warped_im_path='\\fatherserverdw\andreex\students\Donald Monkey fetus\test_registration\registered_BTC_old\elastic registration\FXFAD12FB2_0253.jpg';
original_im_path='\\fatherserverdw\andreex\students\Donald Monkey fetus\test_registration\registered_BTC_old\FXFAD12FB2_0253.jpg';

og_im=imread(original_im_path);
warped_im=imread(warped_im_path);

outpth='\\fatherserverdw\andreex\students\Donald Monkey fetus\test_registration\';

D_pth='\\fatherserverdw\andreex\students\Donald Monkey fetus\test_registration\registered_BTC_old\elastic registration\save warps\D\FXFAD12FB2_0253.mat';
TA=imread('\\fatherserverdw\andreex\students\Donald Monkey fetus\test_registration\TA_old\FXFAD12FB2_0253.tif');
% szz=[6363,12718];
szz=size(warped_im);
padall=0;
fillval=padall;
IHC=2;

imv0=warped_im_path;
TAmv=TA;



matpth = '\\fatherserverdw\andreex\students\Donald Monkey fetus\test_registration\registered_BTC_old\elastic registration\save warps\';
nm='FXFAD12FB2_0253.';



immv0=og_im;
[immv,immvg,TAmv,fillval]=preprocessing(immv0,TAmv,szz,padall,IHC);

disp('   Registration already calculated');disp('')
load([matpth,nm,'mat'],'tform','cent','f');

immvGg=register_global_im(immvg,tform,cent,f,mode(immvg(:)));
TAmvG=register_global_im(TAmv,tform,cent,f,0);


immvE=imwarp(immvG,D,'nearest','FillValues',fillval);
% immvE=imwarp(immvG,D);
imwrite(immvE,outpth);

















