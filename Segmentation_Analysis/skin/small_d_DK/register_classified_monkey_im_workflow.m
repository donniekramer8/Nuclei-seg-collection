
% make TAs

pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\make_giant_confusion_matrix\all_ims_combined\combined classified images\';
imlist = dir([pth, '*.tif']);
outpthTA = [pth,'TA\']; if ~exist(outpthTA, "dir"); mkdir(outpthTA);end

for i=1:length(imlist)
    disp(i)
    nm = imlist(i).name;
    if ~exist([outpthTA,nm],"file")
        im0 = imread([pth, nm]);
        TA = im0~=71 & im0~=0; % WS 
        % TA = TA~=59; % ECM
        imwrite(TA, [outpthTA,nm]);
    else
        disp('SKIPPING')
    end
end


%% IMAGES ARE IN 5X, NEED TO DOWNSAMPLE TO 2.5X

clas_im_pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\make_giant_confusion_matrix\all_ims_combined\combined classified images\';
TA_pth = [clas_im_pth, 'TA\'];

outpth_ds = [clas_im_pth,'ds2\'];
TA_outpth_ds = [outpth_ds, 'TA\'];
if ~exist(outpth_ds, "dir"); mkdir(outpth_ds);end
if ~exist(TA_outpth_ds, "dir"); mkdir(TA_outpth_ds);end

ds=2;

for i=1:length(imlist)
    disp(i)
    nm = imlist(i).name;

    % classified im
    im0 = imread([clas_im_pth, nm]);
    im = im0(1:ds:end,1:ds:end);
    imwrite(im,[outpth_ds,nm])

    % TA
    TA0 = imread([TA_pth, nm]);
    TA = TA0(1:ds:end,1:ds:end);
    imwrite(TA,[TA_outpth_ds,nm])
end




%%


path(path,'\\10.99.68.178\andreex\students\Donald Monkey fetus\codes\register_images\small_d_DK\')

pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\make_giant_confusion_matrix\all_ims_combined\combined classified images\ds2\';

register_images_2023_DK_copy(pth)