%% load in D, scale up to im size, apply inverse of D, then apply new D
% new D calculated in \\10.99.68.178\andreex\students\Donald Monkey fetus\codes\register_images\small_d_DK\FINAL_fix_monkey_reg.m
% after doing some manual adjustments to global reg and also increasing max 
% distance in make_final_grids.m from 75 to 150


%% load paths and make lists of file names
pth_cropped_ims = '\\10.99.68.178\andreex\data\monkey fetus\gestational 40\2_5x\cropped_images\';
pth_TA = [pth_cropped_ims, 'TA\'];
pth_og_save_warps = [pth_cropped_ims 'registered\elastic registration\save_warps\'];
pth_og_D = [pth_og_save_warps, 'D\'];

cropped_ims = dir([pth_cropped_ims, '*.tif']);
og_save_warps = dir([pth_og_save_warps, '*.mat']);
og_D = dir([pth_og_D, '*.mat']);

pth_new_registration = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\';
pth_new_save_warps = [pth_new_registration, 'save_warps\'];
pth_new_D = [pth_new_save_warps, 'D\'];

new_save_warps = dir([pth_new_save_warps, '*.mat']);
new_D = dir([pth_new_D, '*.mat']);

load('\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\registered\elastic registration\save_warps\monkey_fetus_40_0001.mat', "padall", "szz")
IHC=0;

%% start by trying to load monkey_fetus_40_0296.tif and apply global and elastic steps for sanity check
nm = 'monkey_fetus_40_0296';

pth_og_elastic_reg = [pth_cropped_ims 'registered\elastic registration\'];

im_elastic = imread([pth_og_elastic_reg,nm,'.jpg']);

% im_cropped = imread([pth_cropped_ims,nm,'.tif']);
[im_cropped,TA_cropped_im]=get_ims(pth_cropped_ims,nm,'.tif',IHC); % get im and TA

[im_cropped,~,~]=preprocessing(im_cropped,TA_cropped_im,szz,padall,IHC); % pad and format, make grey version of image
imshow(im_cropped)

load([pth_og_save_warps,nm,'.mat'],'tform','cent','f');
fill_val = squeeze(mode(mode(im_cropped,2),1))'; % get fill value for pixels that got warped
im_global_reg=register_global_im(im_cropped,tform,cent,f,fill_val); % global register grey image
imshow(im_global_reg)

load([pth_og_D,nm,'.mat'],'D');
D=imresize(D,size(im_global_reg,1:2)); % scale transformation matrix up to image size

im_elastic_reg = imwarp(im_global_reg,D,'nearest','FillValues',fill_val); % elastic

imshow(im_elastic_reg)

imshowpair(im_elastic_reg, im_elastic)


%% test inverting global reg
reversed_tform = invert(tform);
cent_test = cent;
f_test = abs(0-f);
fill_val_test = fill_val;
im_unglobal_reg = register_global_im(im_global_reg,reversed_tform,cent_test,f_test,fill_val_test);
% imshowpair(im_unglobal_reg, im_global_reg)
imshowpair(im_unglobal_reg, im_cropped)

%% test inverting elastic reg
inv_D = cat(3, -D(:,:,1), -D(:,:,2));

im_unelastic_reg = imwarp(im_elastic_reg,inv_D,'nearest','FillValues',fill_val); % elastic
%imshowpair(im_unelastic_reg,im_elastic_reg)
imshowpair(im_unelastic_reg,im_global_reg)

%% now redo registration and apply new then compare

im_unelastic_reg = imwarp(im_elastic_reg,inv_D,'nearest','FillValues',fill_val); % elastic
im_unglobal_reg = register_global_im(im_unelastic_reg,reversed_tform,cent_test,f_test,fill_val_test);
% imshowpair(im_unglobal_reg,im_cropped)

%%
new_base_im = im_unglobal_reg;


load([pth_new_save_warps,nm,'.mat'],'tform','cent','f');
load([pth_new_D,nm,'.mat'],'D');
im_global_reg=register_global_im(new_base_im,tform,cent,f,fill_val); % global register grey image
D=imresize(D,size(im_global_reg,1:2)); % scale transformation matrix up to image size
im_elastic_reg = imwarp(im_global_reg,D,'nearest','FillValues',fill_val); % elastic
imshow(im_elastic_reg)

im_fixed_elastic = imread([pth_new_registration,nm,'.jpg']);

imshowpair(im_elastic_reg,im_fixed_elastic)

%% try applying to volfinal image
%% get images that were in volfinal
pth_5x = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\make_giant_confusion_matrix\all classifications\classification_MODEL1_3_28_2024_FINAL_v2\registeredE\';
imlist = dir([pth_5x,'*.tif']);

pth_mat = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\dfs_fixed_centroids_3_14\mat_with_reg_coords\w_norm_intensities\';
matlist = dir([pth_mat, '*.mat']);

inds = zeros(length(matlist),1);
mat_counter = 1;
for i=1:length(imlist)
    nm0 = imlist(i).name; nm0 = nm0(1:end-4);
    mat_nm = matlist(mat_counter).name; mat_nm = mat_nm(1:end-4);
    if nm0 == mat_nm
        inds(mat_counter) = i;
        mat_counter = mat_counter + 1;
    end
end

imlist = imlist(inds);

%%
im_name = [nm,'.tif'];
volfinal_ind = find(strcmp(cellstr({imlist.name}), im_name));

%% load volfinal
% load('\\10.162.80.16\Andre_expansion\data\monkey_fetus\final_vols\final_volfinal.mat')

%%
classified_im = volfinal(:,:,volfinal_ind);
imagesc(classified_im)

%% now I have to "uncrop" the classified im by getting rr and padding with 0s till it is right size, then recrop
load('\\10.162.80.16\Andre_expansion\data\monkey_fetus\final_vols\4_3_2024_vols\model 1\mat files\MODEL1_3_28_2024_FINAL_v2.mat','rr')

%%

left_pad = rr(1);
right_pad = size(im_cropped,2)-left_pad-rr(3)-1;

up_pad = rr(2);
down_pad = size(im_cropped,1)-up_pad-rr(4)-1;

pad_fill_val = 0;

padded_classified_im = padarray(classified_im, [0 left_pad], pad_fill_val, 'pre');
padded_classified_im = padarray(padded_classified_im, [0 right_pad], pad_fill_val, 'post');
padded_classified_im = padarray(padded_classified_im, [up_pad 0], pad_fill_val, 'pre');
padded_classified_im = padarray(padded_classified_im, [down_pad 0], pad_fill_val, 'post');

imagesc(padded_classified_im)

%% "unalign" the padded classified im
load([pth_og_D,nm,'.mat'],'D');
load([pth_og_save_warps,nm,'.mat'],'tform','cent','f');
D=imresize(D,size(padded_classified_im,1:2)); % scale transformation matrix up to image size

inv_D = cat(3, -D(:,:,1), -D(:,:,2));
im_unelastic_reg = imwarp(padded_classified_im,inv_D,'nearest','FillValues',pad_fill_val); % elastic

imshowpair(im_unelastic_reg, padded_classified_im)

%%
reversed_tform = invert(tform);
cent_test = cent;
f_test = abs(0-f);
fill_val_test = fill_val;
im_unglobal_reg = register_global_im(im_unelastic_reg,reversed_tform,cent_test,f_test,pad_fill_val);

imagesc(im_unglobal_reg)

%%
load([pth_new_save_warps,nm,'.mat'],'tform','cent','f');
load([pth_new_D,nm,'.mat'],'D');

im_global_reg=register_global_im(im_unglobal_reg,tform,cent,f,pad_fill_val); % global register grey image
D=imresize(D,size(im_global_reg,1:2)); % scale transformation matrix up to image size
im_elastic_reg = imwarp(im_global_reg,D,'nearest','FillValues',pad_fill_val); % elastic
imagesc(im_elastic_reg)

%% recrop
fixed_reg_volfinal_ind = imcrop(im_elastic_reg, rr);

%%
imshowpair(fixed_reg_volfinal_ind, classified_im)

%% now look at next im
next_im = volfinal(:,:,volfinal_ind+3);

% original
figure(1);imshowpair(classified_im, next_im)

% fixed
figure(2);imshowpair(fixed_reg_volfinal_ind, next_im)