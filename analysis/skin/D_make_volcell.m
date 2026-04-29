pth = '\\10.162.80.16\Andre_expansion\data\Skin Lymphedema\L0001_diseased_150\StarDist_9_17_2024_pdac\stardist_feature_df_pickles\xy_inds\registered\mat_with_reg_coords/';
matlist = dir([pth,'*.mat']);

reg_im_pth = '\\10.162.80.16\Andre_expansion\data\Skin Lymphedema\L0001_diseased_150\1x_python\registered\elastic registration\';
im0 = imread([reg_im_pth,'L001_diseased_0001.jpg']);

scale = 2; % 1.25x -> 2.5x

x = size(im0,1)*scale;
y = size(im0,2)*scale;
z = length(matlist);

volcell = zeros(x,y,z,'uint64');

%% load volfinal
pth_volfinal = '\\10.162.80.16\Andre_expansion\data\Skin Lymphedema\mat files_interpolated\L0001_diseased_150_H&E_8_10_2024_ds2_tiled.mat';
load(pth_volfinal)

%%
% make volcell. get the mat file number from z number of volcell.
% get the row number of a cell by looking at its value

load([pth,matlist(1).name], "colnames")

for i=1:length(matlist)
    disp(i)
    matnm = [pth,matlist(i).name];
    load(matnm, 'df')
    xy = df(:,1:2)*scale; % maybe change

    row_nums = 1:length(xy);

    im = zeros(x,y); % make blank im of size x,y
    subscripts = uint64(sub2ind([x,y], uint64(xy(:,2)), uint64(xy(:,1)))); % get subscrits of locations

    im(subscripts) = row_nums;  % label each cell as row num in features table for this slide
    volcell(:,:,i) = im;
   
end

%% save uncropped
pthout = '\\10.162.80.16\Andre_expansion\data\Skin Lymphedema\L0001_diseased_150\StarDist_9_17_2024_pdac\stardist_feature_df_pickles\xy_inds\registered\mat_with_reg_coords\volcell\volcell_uncropped.mat';
save(pthout,"volcell")

%%
% crop the volcell
volcell_cropped = zeros(size(vol),'uint64');

for i=1:size(volcell,3)
    im = volcell(:,:,i);
    im = imcrop(im,rr); % crop however you cropped it, idk what rr means
    volcell(:,:,i) = im;
    disp('')
end