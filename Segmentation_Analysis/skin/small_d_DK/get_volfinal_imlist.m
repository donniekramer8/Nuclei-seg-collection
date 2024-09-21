function imlist=get_volfinal_imlist()
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
end