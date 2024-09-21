%% description
% started 3_7_24

% I want to bin distances into like 5 groups or so, and then do
% PCA on a sample of nuclei from each of those 5 groups to see how they
% compare. I might use only a single class, such as mesoderm for that part
% in order to reduce the compounding variables.

%% code

pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_super_good_version\';

features_table_pth = [pth, 'features_table.mat'];
colnames_pth = [pth, 'colnames_feature_table.mat'];

% table with row number and distance from nearest blood vessels
row_dist_class_tab_pth = [pth,'blood_vessel_distances_old_vol\row_dist_class_tab.mat'];

%% load features table
load(features_table_pth, "features_table");
load(colnames_pth, "colnames");

%% load row/distances tab
load(row_dist_class_tab_pth, "row_dist_class_tab");

%%
% I only want to do this for mesoderm class:
mesoderm = 58;

condition = row_dist_class_tab(:,3) == mesoderm;

mesoderm_row_dist_class_tab = row_dist_class_tab(condition,:);

%% now that I have this table, I need to pull features from features_tab into it

mesoderm_rows = mesoderm_row_dist_class_tab(:,1);

















