path(path,'\\10.99.68.178\andreex\students\Donald Monkey fetus\codes\register_images\small_d_DK\')
path(path,'\\10.99.68.178\andreex\students\Donald Monkey fetus\codes\monkey\')

%%
load('\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\registered\elastic registration\save_warps\monkey_fetus_40_0001.mat', "padall", "szz")

pth = '\\10.99.68.178\andreex\data\monkey fetus\gestational 40\2_5x\cropped_images\';
imlist = dir([pth, '*.tif']);

save_warps = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\registered\elastic registration\save_warps\';
matpth = [save_warps, 'D\'];

ereg_pth = '\\10.99.68.178\andreex\data\monkey fetus\gestational 40\2_5x\cropped_images\registered\elastic registration\';
ereg_imlist = dir([ereg_pth, '*.jpg']);

fix_pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\';


%%

% slides that I identified are poorly aligned based on looking at mostly
% just the ribs.
bad_slides = [257;...
    (289:296)';...
    338;...
    %685;...
    878;...
    879;...
    %880;...
    %882;...
    889;...
    897;...
    898;...
    922;...
    1083;...
    1084;...
    1105];

IHC=0;
rsc=2; % idk what this is
iternum=2; % something for global registration

out_save_warps = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\save_warps\';
out_D = [out_save_warps,'D\'];

if ~exist(out_save_warps, "dir");mkdir(out_save_warps);end
if ~exist(out_D, "dir");mkdir(out_D);end

out_global = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\save_warps\';
out_D = [out_global,'D\'];

%%

fixed_reg_D_pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\save_warps\D\';

% for i=length(imlist)-1:-1:2 % so that it doesn't break at first and last images
for i=1:length(imlist)-1 % so that it doesn't break at first and last images
    moving_name = imlist(i).name;

    % this is important because the registration was calculated starting at
    % middle image and went to ends
    if i > floor(length(imlist)/2)
        %nm1 = imlist(i-1).name; rf1_0 = imread([pth,nm1]);
        %nm2 = imlist(i-2).name; rf2_0 = imread([pth,nm2]);
        %nm3 = imlist(i-3).name; rf3_0 = imread([pth,nm3]);
        %subplot(1,3,1);imshow(rf1_0);subplot(1,3,2);imshow(rf2_0);subplot(1,3,3);imshow(rf3_0)


        reference_name = imlist(i-2).name;
    else
        %nm1 = imlist(i+1).name; rf1_0 = imread([pth,nm1]);
        %nm2 = imlist(i+2).name; rf2_0 = imread([pth,nm2]);
        %nm3 = imlist(i+3).name; rf3_0 = imread([pth,nm3]);

        reference_name = imlist(i+1).name;
    end

    %for j=1:length(bad_slides)-12 % last one is done praise the lord
    for j=11:length(bad_slides) % last one is done praise the lord

        % if the moving image name is in the bad slides defined above
        if contains(moving_name, num2str(bad_slides(j)))
            disp(['moving image: ', moving_name])
            disp(['reference image: ', reference_name])

            % make reference image (global aligned)
            [rf_img,TA_rf_img]=get_ims(pth,reference_name(1:end-4),'.tif',IHC); % get im and TA
            %[~,TA_rf_img,~]=calculate_tissue_space_082_DK(pth,reference_name(1:end-4));
            
            [rf_img,rf_img_grey,TA_rf_img]=preprocessing(rf_img,TA_rf_img,szz,padall,IHC); % pad and format, make grey version of image

            % CHECK***
            %load([save_warps,moving_name(1:end-4),'.mat'],'tform','cent','f'); % load the tform (only global)
            if exist(['\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\save_warps\' reference_name(1:end-4),'.mat'], 'file')
                disp('loading fixed tform')
                load(['\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\save_warps\',reference_name(1:end-4),'.mat'],'tform','cent','f'); % load the tform (only global)
            else
                load([save_warps,reference_name(1:end-4),'.mat'],'tform','cent','f'); % load the tform (only global)
            end

            rf_img_grey_global=register_global_im(rf_img_grey,tform,cent,f,mode(rf_img_grey(:))); % global register grey image
            TA_rf_img_global=register_global_im(TA_rf_img,tform,cent,f,0); % global register the TA


            % make moving image
            [mv_img,TA_mv_img]=get_ims(pth,moving_name(1:end-4),'.tif',IHC); % get im and TA
            %[~,TA_mv_img,~]=calculate_tissue_space_082_DK(pth,moving_name(1:end-4));
            [mv_img,mv_img_grey,TA_mv_img]=preprocessing(mv_img,TA_mv_img,szz,padall,IHC); % pad and format, make grey version of image


            % calculate global registration between mv img and global ref im
            % f = 0;
            %[mv_img_grey_global,tform,cent,R]=calculate_global_reg_DK_new(rf_img_grey_global,mv_img_grey,rsc,iternum,IHC);
            [mv_img_grey_global,tform,cent,f,R]=calculate_global_reg_original(rf_img_grey_global,mv_img_grey,rsc,iternum,IHC);
            TA_mv_img_global=register_global_im(TA_mv_img,tform,cent,f,IHC); % global register TA image
            

            % STOP HERE*** tweak elastic settings
            num = .25; % just used for changing below ratios
            regE.szE=250/num; % size of registration tiles % 250
            regE.bfE=100/num; % size of buffer on registration tiles 100
            regE.diE=200/num; % distance between tiles     % 150  200

            regE.szE=400; % size of registration tiles % 250
            regE.bfE=100; % size of buffer on registration tiles 100
            regE.diE=200; % distance between tiles     % 150  200

            % get elastic registration between global aligned rf and global aligned mv
            Dmv=calculate_elastic_registration(rf_img_grey_global,mv_img_grey_global,TA_rf_img_global,TA_mv_img_global,regE.szE,regE.bfE,regE.diE);
            
            % add that to the previous transformation matrix and then upscale
            if exist([fixed_reg_D_pth reference_name(1:end-3),'mat'], 'file')
                load([fixed_reg_D_pth reference_name(1:end-3),'mat'],'D'); D_og = D;
            else
                load([matpth,reference_name(1:end-3),'mat'],'D'); D_og = D;
            end
            D=D+Dmv;
            D=imresize(D,size(mv_img_grey)); % scale transformation matrix up to image size
            fillval=squeeze(mode(mode(mv_img,2),1))'; % get fill value for pixels that got warped

            % all that stuff was done on grey scale image, so this applies the transformation matrix to the rgb image
            % make output image
            imout = register_global_im(mv_img,tform,cent,f,fillval); % global
            imout = imwarp(imout,D,'nearest','FillValues',fillval); % elastic

            % load ref im that was already registered
            if exist(['\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\' reference_name(1:end-3),'jpg'], 'file')
                disp('reading fixed')
                e_ref_im = imread(['\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\' reference_name(1:end-3),'jpg']);
            else
                e_ref_im = imread([ereg_pth,reference_name(1:end-3),'jpg']);
            end
            figure(68);imshowpair(imout,e_ref_im)
            disp('')

            imwrite(imout,['\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\' [moving_name(1:end-3),'jpg']])
            save([out_global,moving_name(1:end-3),'mat'],'tform','cent','szz','padall','regE','f');
            D = imresize(D, size(D_og,1:2));
            save([out_D,moving_name(1:end-3),'mat'],'D');

        end
    end

end





