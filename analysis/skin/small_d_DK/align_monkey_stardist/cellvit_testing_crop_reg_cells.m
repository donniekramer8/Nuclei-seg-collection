% pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_coords_with_features\';
pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\dfs_fixed_centroids_3_14\mat_with_reg_coords\w_norm_intensities\';
matlist = dir([pth, '*.mat']);

sz = size(imread('\\10.99.68.178\andreex\data\monkey fetus\gestational 40\2_5x\cropped_images\registered\elastic registration\monkey_fetus_40_0001.jpg'));
%sz=[rr(4)+1,rr(3)+1];
reg_im_pth = '\\10.99.68.178\andreex\data\monkey fetus\gestational 40\2_5x\cropped_images\registered\elastic registration\';

% load('\\10.162.80.16\Andre_expansion\data\monkey_fetus\vols\vol_all_combined_8_24_2023.mat', "rr")
% load('\\169.254.138.20\Andre_expansion\data\monkey_fetus\final_vols\4_3_2024_vols\all_models\cmap_titles\cmap_titles_4_9_2024.mat')
load('\\10.162.80.16\Andre_expansion\data\monkey_fetus\final_vols\4_3_2024_vols\model 1\mat files\MODEL1_3_28_2024_FINAL_v2.mat','rr')

sz_crop = [(rr(4)+1), (rr(3)+1)];
sz=sz(1:2);
% sz=sz_crop;

volcell = zeros([sz_crop,length(matlist)], 'single');

%% load giant table (takes long time, beware)
pth_features_table = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_super_good_version\';
% load([pth_features_table, 'features_table.mat'], "features_table")
% load([pth_features_table, 'features_table_w_norms.mat'], "features_table")
load([pth_features_table, 'features_table_w_norms_cropped_reg_xy_fixed_4_11.mat'], "features_table")
% load([pth_features_table, 'colnames_feature_table.mat'], "colnames")

%


%% make volcell and instead of 0 and 1s, do 0 for no cell, row number for cell
% (row number in features_table)
volcell = zeros([sz_crop,length(matlist)], 'single');
total_n_unique = 0;
slide_numbers = unique(features_table(:,end));
xy_counter = 0;
in_vol_counter = 1;  % 1 instead of 0 because accounting for 0 in volcell
% start_ind = 1;
% figure, imagesc(volM1s(:,:,500)), hold on,scatter(df(:,1)-87,df(:,2)-24)

%pad = 800;


for i=1:length(matlist)
    disp(matlist(i).name)

    slide_no = slide_numbers(i);
    start_ind = find(features_table(:, 22) == slide_no, 1);

    % subset df
    df = features_table(features_table(:,end) == slide_no,:);
    xy = round(df(:,1:2));


    %xy = [xy(:,1)+rr(1), xy(:,2)+rr(2)];

    [numRows, ~] = size(df);
    end_ind = start_ind + numRows - 1;  % idk why i have to -1 it won't work without it tho

    valid_indices = xy(:, 1) >= 1 & xy(:, 1) <= sz(2) & xy(:, 2) >= 1 & xy(:, 2) <= sz(1);
    xy_og = xy;
    xy = xy(valid_indices, :);

    row_num_inds = start_ind:end_ind;
    row_num_inds = row_num_inds(valid_indices);  % get rid of ones that aren't in crop

    im = zeros(sz);
    subscripts = sub2ind(size(im), xy(:,2), xy(:,1));

    im(subscripts) = row_num_inds;  % label each cell as row num in features table

    im = imcrop(im, rr);
    volcell(:,:,i) = im;

    ind_p = length(unique(im)) - 1;  % -1 because of the 0 val

    disp(['unique xy_og: ', num2str(length(unique(xy_og, 'rows')))]);
    disp(['unique xy: ', num2str(length(unique(xy, 'rows')))]);
    disp(['# unique in im: ', num2str(ind_p)])
    disp(['# cells: ', num2str(length(unique(im))/length(unique(xy, 'rows')))])

    xy_counter = xy_counter + length(xy);
    in_vol_counter = in_vol_counter + ind_p;

    disp(['Start ind: ', num2str(start_ind)])
    disp(xy_counter)
    disp(in_vol_counter)

    disp('')

    %if (i>130)
    if (i>35)
        disp('')
    end
    start_ind = end_ind + 1;

end

%%
% volcell = uint8(volcell);  % didn't run this first time on 2_28_24

outpth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_super_good_version\';
% outpth = '\\169.254.138.20\Andre_expansion\data\monkey_fetus\final_vols\4_3_2024_vols\all_models\';
% save([outpth,'volcell_linked_v2.mat'], "volcell", "rr", "sz", "pth", "-v7.3")
save([outpth,'volcell_IDs_FINAL.mat'], "volcell", "rr", "sz", "pth", "-v7.3")

