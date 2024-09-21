pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_coords_with_features\';
matlist = dir([pth, '*.mat']);

sz = size(imread('\\10.99.68.178\andreex\data\monkey fetus\gestational 40\2_5x\cropped_images\registered\elastic registration\monkey_fetus_40_0001.jpg'));
% pad_amt = 200;
% sz = [sz(1)+pad_amt, sz(2)+pad_amt];
sz = sz(1:2);

reg_im_pth = '\\10.99.68.178\andreex\data\monkey fetus\gestational 40\2_5x\cropped_images\registered\elastic registration\';



load('\\10.162.80.16\Andre_expansion\data\monkey_fetus\vols\vol_all_combined_8_24_2023.mat', "rr")

sz_crop = [(rr(4)+1), (rr(3)+1)];


n_features =  1; %16;

volcell = zeros([sz_crop,length(matlist)], 'single');

%% load giant table (takes long time, beware)
pth_features_table = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\Stardist\StarDist_12_25_23\volcell_super_good_version\';
% load([pth_features_table, 'features_table.mat'], "features_table")
load([pth_features_table, 'features_table_w_norms_cropped_reg_xy_fixed.mat'], "features_table")
load([pth_features_table, 'colnames_feature_table.mat'], "colnames")


%% make volcell and instead of 0 and 1s, do 0 for no cell, row number for cell
% (row number in features_table)

volcell = zeros([sz_crop,length(matlist)], 'single');
% for debugging only
% gt = zeros([50,1]);
% after_ds = zeros([50,1]);
total_n_unique = 0;

slide_numbers = unique(features_table(:,end));

xy_counter = 0;
in_vol_counter = 1;  % 1 instead of 0 because accounting for 0 in volcell

start_ind = 1;

for i=1:length(matlist)
    disp(matlist(i).name)

    slide_no = slide_numbers(i);

    % subset df
    df = features_table(features_table(:,end) == slide_no,:);
    xy = round(df(:,1:2));
    [numRows, ~] = size(df);
    end_ind = start_ind + numRows - 1;  % idk why i have to -1 it won't work without it tho

    valid_indices = xy(:, 1) >= 1 & xy(:, 1) <= sz(2) & xy(:, 2) >= 1 & xy(:, 2) <= sz(1);
    xy_og = xy;
    xy = xy(valid_indices, :);

    row_num_inds = start_ind:end_ind;
    row_num_inds = row_num_inds(valid_indices);  % get rid of ones that aren't in crop

    im = zeros(sz);
    subscripts = sub2ind(sz, xy(:,2), xy(:,1));

    im(subscripts) = row_num_inds;  % label each cell as row num in features table

    im = imcrop(im, rr);
    volcell(:,:,i) = im;

    figure(4); imagesc(volfinal(:,:,i))
    figure(5); imagesc(im>0)

    %p = length(unique(volcell(:,:,1:i)));
    ind_p = length(unique(im)) - 1;  % -1 because of the 0 val
    %disp(p)
    %total_n_unique = total_n_unique + ind_p - 1; % -1 because 0 is also unique in im
    %disp(total_n_unique)


    disp(['unique xy_og: ', num2str(length(unique(xy_og, 'rows')))]);
    disp(['unique xy: ', num2str(length(unique(xy, 'rows')))]);
    disp(['# unique in im: ', num2str(ind_p)])
    disp(['# cells: ', num2str(length(unique(im))/length(unique(xy, 'rows')))])
    %disp('')
    % 
    % gt(i) = length(unique(xy_og, 'rows'));
    % after_ds(i) = length(unique(xy, 'rows'));

    % imshow(im);
    % reg_im = imread([reg_im_pth, matlist(i).name(1:end-4),'.jpg']);
    % figure(2); imshowpair(reg_im, im);

    % debugging
    % temp = im(im > 0);
    % temp_df = features_table(temp,:);
    % disp(unique(temp_df(:,end)))

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
% save([outpth,'volcell_linked_v2.mat'], "volcell", "rr", "sz", "pth", "-v7.3")

