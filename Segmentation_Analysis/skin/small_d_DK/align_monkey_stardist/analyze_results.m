%%
% started 3/6/24
% since I am giving up on python since it is slow and the 0-1 indexing
% thing upset me after yesterday, I am going to try to do the analysis in
% matlab now.

% this code's initial purpose is to try to develop a workflow for linking
% the volcell with hte features_table in an efficient manner. However,
% hopefully at the end I will be able to do something cool with batching
% the cells based on how far away they are from blood vessels or notochord,
% etc. and then seeing if they cluster based on distance away from
% morphology based on morphology features in big table.


%% define paths
pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_super_good_version\';
% features table
pth_features_table = [pth, 'features_table_w_norms.mat'];
% col names in features table
pth_colnames = [pth, 'colnames_feature_table.mat'];
% volcell that is linked to features_table
pth_volcell = [pth, 'volcell_linked_v2.mat'];

%% load rr just in case, probably won't need
load('\\10.162.80.16\Andre_expansion\data\monkey_fetus\vols\vol_all_combined_8_24_2023.mat', "rr")

%% load features_table and volcell
load(pth_features_table, "features_table");
load(pth_colnames, "colnames");
load(pth_volcell, "volcell");

%% Begin analysis

%% For a single slide, subset the df based on the values in volcell at that slide

z_num = 500;
slide = volcell(:,:,z_num);

df_inds = slide > 0;

subset_df = features_table(df_inds, :);









%% ideas for blood vessel experiment:
% make a new volcell, 0 = no cell, nonzero = cell's distance to blood
% vessel

% then, basically add that as a feature to features_table, maybe don't have
% to exactly do it that way

% then subset features_table by binned distances

% take sample of x from each subsetted df and do pca on features (besides
% distance from blood vessel obviously

% how to get distance: just do bins in the first place as feature:
% take blood vessel vol, then dilate the vol by x voxels each iteration
% until its all full. each iteration, go thru each cell remaining, if cell
% not in expanding blood vessel volume yet, but now is, then say it is in
% that binned distance.

% make copy of volcell. Set nonzero values to like -1 or something. If at
% end, any zeros remain, then just say it is the max binned distance or
% something.

% then update the values each iteration

%% get distance away from blood vessels for each cell

%% load volfinal
% need to update with new final vol that Ashley is making rn
pth_volfinal = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\final_vols\vol_all_combined_10_16_clean.mat';

load(pth_volfinal, 'volfinal');

%% subset volfinal to be only blood vessel classes and make binary

bv_classes = [23, 25:29, 31, 38, 59];

% new ones I think, double check though [26 29:33 36 44 67];

vol_bv = volfinal;
vol_bv(~ismember(volfinal, bv_classes)) = 0;

vol_bv(vol_bv > 0) = 1;

%% test dilation on downsampled vol
ds_vol_bv = vol_bv(1:32:end, 1:32:end, 1:32:end);

radius = 50;

se = strel('disk',radius);
ds_vol_bv = imdilate(ds_vol_bv, se); % Dilate the binary volume
volshow(ds_vol_bv)

%% imdilate the vol

vol_bv_copy = vol_bv(1:16:end,1:16:end,1:16:end);

% clear volfinal
radius = 2;

se = strel('sphere',radius);

vol_bv_copy = imdilate(vol_bv_copy, strel('sphere',radius)); % Dilate the binary volume

distances=bwdist(vol_bv);

%% SMALLSCALE BWDIST

distances=bwdist(vol_bv);


%% Pseudo code

% load volcell
% linearize volcell 
% iterate for each cell
% bwdist based on that cell (a lot faster cause its one point)
% multiply bwdist with bv_vol
% get min value in the bwdist .* bv_vol - this is trhe distance to the
% closes bv
















