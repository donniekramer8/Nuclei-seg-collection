%% description
% started 3_7_24
% this code is meant to take the volcell from the monkey fetus, where each
% nonzero value represents a row in the feature table... and link that with
% the distances volcell that I made in make_distance_vol.m

% note, towards the bottom, where I make the row_dist_class_tab matrix,
% the code starts to get a little messy. It is important to make the table
% at least int32 in order to keep the row values from hitting the max int
% number with int16. It also can't be unsigned because some distances are
% still -1 in row_dist_class_tab, this could now be updated to be 0 since
% it doesn't really need to be -1 anymore, the only reason I did that was
% to make sure not to lose those cells. A good sanity check is to make sure
% that all -1 dist cells are also a blood vessel class.


%% code

pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_super_good_version\';

volcell_pth = [pth, 'volcell_linked_v2.mat'];
bv_dist_volcell_pth = [pth,'blood_vessel_distances_old_vol\bv_dist_volcell_old.mat'];

features_table_pth = [pth, 'features_table.mat'];
colnames_pth = [pth, 'colnames_feature_table.mat'];


%% load bv_dist_volcell
load(bv_dist_volcell_pth, "bv_dist_volcell");

%% load features table
load(features_table_pth, "features_table");
load(colnames_pth, "colnames");

%% get non_zero indices of bv_dist_volcell
bv_dist_inds = bv_dist_volcell ~= 0;

% this is the total number of nuclei within the set of slides that were not
% cropped out I think
sum(sum(sum(bv_dist_inds)))

% sum(sum(sum(volcell))) = 421806569

%% linearize those indices into a smaller list or something
inds = find(bv_dist_inds == 1);

%% volcell(inds) = rows of the the table
distances = bv_dist_volcell(inds);

%% load volcell
% clear bv_dist_volcell
load([pth, 'volcell_linked_v2.mat'], "volcell");

%%
rows = volcell(inds);

%%
row_dist_tab = [rows, uint32(distances)];

outpth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_super_good_version\blood_vessel_distances_old_vol\row_dist_tab.mat';
save(outpth, "row_dist_tab")




















%% I need to get classes of cells in the row in that table as well, from volfinal

%% load volfinal

% need to update with new final vol
pth_volfinal = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\final_vols\vol_all_combined_10_16_clean.mat';

load(pth_volfinal, 'volfinal');

%% Remove unused slides from volfinal that are not in volcell

% these are slide names that I used to make volcell:
volcell_files_pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\dfs_fixed_centroids_3_14\mat_with_reg_coords\w_norm_intensities\';
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

%%
% I accidentally cleared inds from memory, but this should also work and be
% faster if you load volcell
inds = find(volcell > 0);
classes = volfinal(inds);
%%
row_dist_class_tab = [int32(row_dist_tab), classes];
%%
outpth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_super_good_version\blood_vessel_distances_old_vol\row_dist_class_tab.mat';
save(outpth, "row_dist_class_tab", '-v7.3')




%% ignore this part 
temp = int32(row_dist_class_tab);
temp2 = [rows, temp(:,2:3)];

save("\\10.162.80.16\Andre_expansion\data\monkey_fetus\final_vols\4_3_2024_vols\all_models\volfinal_volcell_4_11_24\volfinal.mat", "volfinal", '-v7.3');
save("\\10.162.80.16\Andre_expansion\data\monkey_fetus\final_vols\4_3_2024_vols\all_models\volfinal_volcell_4_11_24\volcell.mat", "volcell", '-v7.3');
save("\\10.162.80.16\Andre_expansion\data\monkey_fetus\final_vols\4_3_2024_vols\all_models\volfinal_volcell_4_11_24\other_data.mat", "volcell_files", "rr", "sz", "sxz", "sk", "scaleHE", "ecm_class")






