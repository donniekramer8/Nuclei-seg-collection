%% fix the poorly registered images in volfinal by unaligning and then applying new registration
path(path, '\\10.99.68.178\andreex\students\Donald Monkey fetus\codes\monkey\')
path(path, '\\10.99.68.178\andreex\students\Donald Monkey fetus\codes\register_images\small_d_DK\')

%%

load('\\10.162.80.16\Andre_expansion\data\monkey_fetus\final_vols\4_3_2024_vols\model 1\mat files\MODEL1_3_28_2024_FINAL_v2.mat','rr')
load('\\10.162.80.16\Andre_expansion\data\monkey_fetus\final_vols\final_volfinal.mat')

%%
pth_cropped_ims = '\\10.99.68.178\andreex\data\monkey fetus\gestational 40\2_5x\cropped_images\';
pth_TA = [pth_cropped_ims, 'TA\'];
pth_og_save_warps = [pth_cropped_ims 'registered\elastic registration\save_warps\'];
pth_og_D = [pth_og_save_warps, 'D\'];

pth_new_registration = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\';
pth_new_save_warps = [pth_new_registration, 'save_warps\'];
pth_new_D = [pth_new_save_warps, 'D\'];

load('\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\registered\elastic registration\save_warps\monkey_fetus_40_0001.mat', "padall", "szz")
IHC=0;
volfinal_imlist = get_volfinal_imlist();

bad_slides = [257;...
    (289:296)';...
    338;...
    % 685;...
    878;...
    %880;...
    %882;...
    889;...
    897;...
    898;...
    922;...
    1083;...
    1084;...
    1105];

pad_fill_val = 0;

im = imread('\\10.99.68.178\andreex\data\monkey fetus\gestational 40\2_5x\cropped_images\registered\elastic registration\monkey_fetus_40_0001.jpg');
uncropped_size = size(im); clear im;

left_pad = rr(1);
right_pad = uncropped_size(2)-left_pad-rr(3)-1;

up_pad = rr(2);
down_pad = uncropped_size(1)-up_pad-rr(4)-1;

%%

outpth_before_after_ims = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\final_vols\rib_segmentation_6_12\fix_registration_images\before_after\';

volfinal2=volfinal;

for i=1:length(bad_slides)
    if bad_slides(i) < 1000  % so that the name of the files are correct
        nm = ['monkey_fetus_40_0', num2str(bad_slides(i))];
    else
        nm = ['monkey_fetus_40_', num2str(bad_slides(i))];
    end

    disp(nm)
    
    % find which z slice of volfinal this image is
    volfinal_ind = find(strcmp(cellstr({volfinal_imlist.name}), [nm, '.tif']));
    
    % read z slice of volfinal
    im0 = volfinal(:,:,volfinal_ind);

    % pad im to get back to uncropped version so that it can be same size
    % as the elastic reg matrix (D)
    padded_im0 = padarray(im0, [0 left_pad], pad_fill_val, 'pre');
    padded_im0 = padarray(padded_im0, [0 right_pad], pad_fill_val, 'post');
    padded_im0 = padarray(padded_im0, [up_pad 0], pad_fill_val, 'pre');
    padded_im0 = padarray(padded_im0, [down_pad 0], pad_fill_val, 'post');

    % load original global and elastic variables
    load([pth_og_D,nm,'.mat'],'D');
    load([pth_og_save_warps,nm,'.mat'],'tform','cent','f');

    D=imresize(D,size(padded_im0)); % scale transformation matrix up to image size
    inv_D = cat(3, -D(:,:,1), -D(:,:,2)); % make xy of elastic reg matrix opposite sign

    % undo elastic registration
    im_unelastic_reg = imwarp(padded_im0,inv_D,'nearest','FillValues',pad_fill_val);

    % set up reversed global registration variables
    reversed_tform = invert(tform);
    cent_rev = cent; % idk why this is same as cent but to lazy to look into it
    f_rev = abs(0-f);

    % undo global registration
    im_unglobal_reg = register_global_im(im_unelastic_reg,reversed_tform,cent_rev,f_rev,pad_fill_val);

    clear D tform cent f reversed_tform cent_rev f_rev; % clear variables (not really needed)

    % load new, better registration variables in that were made with this
    % script: \\10.99.68.178\andreex\students\Donald Monkey fetus\codes\register_images\small_d_DK\FINAL_fix_monkey_reg.m
    % in there, some of the images had a semi-manual global step where I
    % changed either the xy translation manually or the angle so that the
    % ribs matched up basically. Also, the elastic step was changed, such
    % that the maximun distance for a tile was increased up to 150 from 75,
    % and also the tiles were bigger and maybe had more overlap
    load([pth_new_save_warps,nm,'.mat'],'tform','cent','f');
    load([pth_new_D,nm,'.mat'],'D');
    
    im_global_reg=register_global_im(im_unglobal_reg,tform,cent,f,pad_fill_val); % global register grey image
    D=imresize(D,size(im_global_reg)); % scale transformation matrix up to image size
    im_elastic_reg = imwarp(im_global_reg,D,'nearest','FillValues',pad_fill_val); % elastic

    clear D tform cent f; % clear variables (not really needed)

    output_im = imcrop(im_elastic_reg, rr); % recrop

    % imagesc(output_im)
    % figure(2);imshowpair(output_im, im0)
    next_im_new = volfinal2(:,:,volfinal_ind-1);
    next_im_og = volfinal(:,:,volfinal_ind-1);
    figure(3);subplot(1,2,1);imshowpair(im0, next_im_og);title(['Old: ' nm(end-3:end)]);
    subplot(1,2,2);imshowpair(output_im, next_im_new);title(['New: ' nm(end-3:end)]);
    set(gcf, 'Position', get(0, 'Screensize'));
    saveas(gcf, [outpth_before_after_ims,nm,'.jpg']);
    % figure(3);imshowpair(im0, next_im)
    % figure(3);imshowpair(output_im, next_im)

    disp('')

    volfinal2(:,:,volfinal_ind) = output_im;
end

%% save
% save('\\10.162.80.16\Andre_expansion\data\monkey_fetus\final_vols\fix_reg_volfinal_v1.mat', "volfinal");

%% 

ribs_nums = 15:38;

ribs_vol = zeros(size(volfinal), 'logical');

for i=1:length(ribs_nums)
    disp(i)
    num = ribs_nums(i);
    vol = volfinal2==num;
    ribs_vol(vol) = 1;
end

%%
ribs_nums = 15:38;

ribs_vol_og = zeros(size(volfinal), 'logical');

for i=1:length(ribs_nums)
    disp(i)
    num = ribs_nums(i);
    vol = volfinal==num;
    ribs_vol_og(vol) = 1;
end

%%
for i=1:4:size(ribs_vol,3)
    disp(i)
    imagesc(ribs_vol(:,:,i))
    disp('')
end
