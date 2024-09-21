%% description
% this code makes a volume from volfinal that has values for each voxel of
% distance to closest blood vessel. This is then .* with volcell to make a
% list of distances for each cell from nearest blood vessel. This will be
% used as an additional featuure in the features table probably for future
% analysis.

%% READ PLEASE
% next time I do this, make sure vol_bv matches up in size w volcell***


%% load volfinal

% need to update with new final vol
pth_volfinal = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\final_vols\vol_all_combined_10_16_clean.mat';

load(pth_volfinal, 'volfinal');

%% Remove unused slides from volfinal that are not in volcell

% these are slide names that I used to make volcell:
volcell_files_pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_coords_with_features\';
volcell_files = dir([volcell_files_pth, '*.mat']);

% these are names of files that Andre used to make volfinal
volfinal_files_pth = '\\10.99.68.178\andreex\data\monkey fetus\gestational 40\5x\cropped_images\';
volfinal_files = dir([volfinal_files_pth, '*.tif']);

% slides to keep in volfinal:
good_slides = zeros(length(volcell_files), 1);

volcell_counter = 1;
for i=1:length(volfinal_files)
    % name of files without extension
    nm_vfinal = volfinal_files(i).name(1:end-4);
    nm_vcell = volcell_files(volcell_counter).name(1:end-4);

    if nm_vfinal == nm_vcell
        good_slides(volcell_counter) = i;
        volcell_counter = volcell_counter + 1;
    end


    disp('')
end

%% now I have good indices of volfinal that match up with volcell

volfinal = volfinal(:,:,good_slides);

%% make blood vessels logical volume from volfinal

bv_classes = [23, 25:29, 31, 38, 59];

% new ones I think, double check though [26 29:33 36 44 67];

vol_bv = volfinal;
vol_bv(~ismember(volfinal, bv_classes)) = 0;

vol_bv(vol_bv > 0) = 1;

%% bwdist -> get distance from nearest bv for each voxel in vol_bv

dist_vol=bwdist(vol_bv);


%% make dist_vol uint8 by rounding or something like that, saves space
dist_vol=int16(dist_vol);

% make 0 -> -1 (turn blood vessel nuc into -1 so that when I .* with
% volcell it won't get rid of those nuc unintentionally
dist_vol(dist_vol == 0) = -1;


%% save dist_vol_uint8


%% load volcell
% clear volfinal

pth_volcell = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_super_good_version\volcell_linked_v2.mat';
load(pth_volcell, "volcell");

%%
volcell = (volcell>0);  % get logical map of volcell
volcell=int16(volcell);
bv_dist_volcell = volcell.*dist_vol;

%% validate:
bv_dist_inds = bv_dist_volcell ~= 0;
sum(sum(sum(bv_dist_inds)))

%%
save('\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_super_good_version\blood_vessel_distances_old_vol\bv_dist_volcell_old.mat', "bv_dist_volcell", '-v7.3')


