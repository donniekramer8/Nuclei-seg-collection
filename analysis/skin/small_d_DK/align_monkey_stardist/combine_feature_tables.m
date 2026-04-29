%pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_coords_with_features\';
% new dfs with normalized staining intensities below
pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\dfs_fixed_centroids_3_14\mat_with_reg_coords\w_norm_intensities\';
matlist = dir([pth, '*.mat']);

% This code is going to take each individual features df for each slide of 
% the monkey and make a big table out of it. 

%% Count number of rows that I need to add to my empty big table

n_rows = 0;

for i=1:length(matlist)
    disp(i)
    load([pth, matlist(i).name], "df", "colnames")
    [numRows, ~] = size(df);

    n_rows = n_rows + numRows;
end

%%
[~, ncol] = size(df);
ncol = ncol;
bigtable = zeros([n_rows, ncol]);

%%

start_ind = 1;

for i=1:length(matlist)
    disp(i)
    load([pth, matlist(i).name], "df", "colnames")
    [numRows, ~] = size(df);

    end_ind = start_ind + numRows - 1;  % idk why i have to -1 it won't work without it tho

    slide_num = str2double(matlist(i).name(17:end-4));  % 4 digit number at end of slide

    %col = repmat(slide_num,numRows, 1);  % add slide num to new column
    %df(:,21) = col;

    bigtable(start_ind:end_ind,:) = df;

    disp(sum(bigtable(:,end) == slide_num))
    disp(length(df))

    start_ind = end_ind + 1;
    disp('')
end

%% get rid of empty rows at end of table
nonZeroRows=all(bigtable,2);
bigtable = bigtable(1:end_ind, :);

% I didn't run this first bc I ran out of memory, should try again

%%

outpth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_super_good_version\';

features_table = single(bigtable);
features_table = features_table(1:end_ind, :);

save([outpth, 'features_table_w_norms_cropped_reg_xy_fixed_4_11.mat'], "features_table", "-v7.3");

% save([outpth, 'colnames_feature_table.mat'], "colnames");

