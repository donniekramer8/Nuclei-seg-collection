path(path, '\\10.99.68.178\andreex\students\Donald Monkey fetus\codes\register_images\small_d_DK\')

% pth of all the data
pth0 = '\\10.162.80.16\Andre_expansion\data\Skin Lymphedema\L0001_diseased_150\';

% pth with registration
pth_reg = [pth0, '1x_python\'];

% pth where xy inds are stored from stardist output after running 1_make_xy_inds.ipynb
pthcoords = [pth0, 'StarDist_9_17_2024_pdac\stardist_feature_df_pickles\xy_inds\'];

outpth = [pthcoords, 'registered\'];
scale = 2; % 1.25x registration to 2.5x resolution

register_cell_coordinates_2024_DK(pth_reg,pthcoords,scale, outpth)