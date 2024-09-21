path(path,'\\10.99.68.178\andreex\students\Donald Monkey fetus\codes\register_images\small_d_DK\')
path(path,'\\10.99.68.178\andreex\students\Donald Monkey fetus\codes\monkey\')

load('\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\registered\elastic registration\save_warps\monkey_fetus_40_0001.mat', "padall", "szz")

pth = '\\10.99.68.178\andreex\data\monkey fetus\gestational 40\2_5x\cropped_images\';
imlist = dir([pth, '*.tif']);

matpth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\registered\elastic registration\save_warps\D\';
save_warps = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\registered\elastic registration\save_warps\';

ereg_pth = '\\10.99.68.178\andreex\data\monkey fetus\gestational 40\2_5x\cropped_images\registered\elastic registration\';
ereg_imlist = dir([ereg_pth, '*.jpg']);

fix_pth = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\';


% slides that I identified are poorly aligned based on looking at mostly
% just the ribs.
bad_slides = [257;...
    338;...
    (289:296)';...
    695;...
    878;...
    880;...
    882;...
    889;...
    897;...
    898;...
    922;...
    1083;...
    1084;...
    1105];


% elastic registration settings
regE.szE=250*2; % size of registration tiles % 250
regE.bfE=100*2; % size of buffer on registration tiles 100
regE.diE=150*2; % distance between tiles     % 150  200

IHC=0;
rsc=2; % idk what this is
iternum=5; % something for global registration

out_global = '\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\save_warps\';
out_D = [out_global,'D\'];

if ~exist(out_global, "dir");mkdir(out_global);end
if ~exist(out_D, "dir");mkdir(out_D);end

%%


for i=2:length(imlist)-1 % so that it doesn't break at first and last images
    moving_name = imlist(i).name;
    %disp(moving_name)

    % this is important because the registration was calculated starting at
    % middle image and went to ends
    if i > floor(length(imlist)/2)
        reference_name = imlist(i-1).name;  %% CHANGE BACK TO -1 !!!
    else
        reference_name = imlist(i+3).name;
    end
    for j=10:length(bad_slides)
        % if the moving image name is in the bad slides defined above
        if contains(moving_name, num2str(bad_slides(j)))

            disp(['moving image: ', moving_name])
            disp(['reference image: ', reference_name])

            % reference image
            [rf_img,TA_rf_img]=get_ims(pth,reference_name(1:end-4),'.tif',IHC); % get im and TA
            % get other TA
            %[~,TA_rf_img,~]=calculate_tissue_space_082_DK(pth,reference_name(1:end-4));
            [rf_img,rf_img_grey,TA_rf_img]=preprocessing(rf_img,TA_rf_img,szz,padall,IHC); % pad and format, make grey version of image

            % load([save_warps,reference_name(1:end-4),'.mat'],'tform','cent','f'); % load the tform (only global)
            load([save_warps,moving_name(1:end-4),'.mat'],'tform','cent','f'); % load the tform (only global)
            %f = abs(f-1);
            rf_img_grey_global=register_global_im(rf_img_grey,tform,cent,f,mode(rf_img_grey(:))); % global register grey image
            TA_rf_img_global=register_global_im(TA_rf_img,tform,cent,f,0); % global register the TA

            % moving image
            [mv_img,TA_mv_img]=get_ims(pth,moving_name(1:end-4),'.tif',IHC); % get im and TA
            % get other TA
            %[~,TA_mv_img,~]=calculate_tissue_space_082_DK(pth,moving_name(1:end-4));
            [mv_img,mv_img_grey,TA_mv_img]=preprocessing(mv_img,TA_mv_img,szz,padall,IHC); % pad and format, make grey version of image
            disp('')

            % calculate global registration
            %f = 0;
            %[mv_img_grey_global,tform,cent,R]=calculate_global_reg_DK_new(rf_img_grey_global,mv_img_grey,rsc,iternum,IHC);
            [mv_img_grey_global,tform,cent,f,R]=calculate_global_reg(rf_img_grey_global,mv_img_grey,rsc,iternum,IHC);
            TA_mv_img_global=register_global_im(TA_mv_img,tform,cent,f,IHC); % global register TA image

            % tweak elastic settings
            num = .25; % just used for changing below ratios
            regE.szE=250/num; % size of registration tiles % 250
            regE.bfE=100/num; % size of buffer on registration tiles 100
            regE.diE=200/num; % distance between tiles     % 150  200

            % try normalizing rf and mv images?
            % figure(80)
            % subplot(1,2,1); imshow(rf_img_grey_global, []); title('Reference image');
            % subplot(1,2,2); imshow(mv_img_grey_global, []); title('Moving image');

            normalize = 1;
            if normalize
                % get mean values of non-zero (within TA) values for rf and mv ims
                median_rf_img_grey_global = median(rf_img_grey_global(rf_img_grey_global(:)~=0));
                median_mv_img_grey_global = median(mv_img_grey_global(mv_img_grey_global(:)~=0));
    
                if median_rf_img_grey_global > median_mv_img_grey_global
                    ratio = double(median_mv_img_grey_global)/double(median_rf_img_grey_global);
                    norm_rf_img_grey_global = uint8(double(rf_img_grey_global)*ratio);
                else
                    ratio = double(median_rf_img_grey_global)/double(median_mv_img_grey_global);
                    norm_rf_img_grey_global = uint8(double(rf_img_grey_global)*ratio);
                end

                % figure(81)
                % subplot(1,2,1); imshow(norm_rf_img_grey_global, [0 255]); title('Norm Reference image');
                %subplot(1,2,2); imshow(rf_img_grey_global, [0 255]); title('Reference image');
                % subplot(1,2,2); imshow(mv_img_grey_global, [0 255]); title('Moving image');
            end

            % just for testing
            % figure(98);histogram(norm_rf_img_grey_global_zs(norm_rf_img_grey_global_zs~=0),10)
            % figure(99);histogram(norm_mv_img_grey_global_zs(norm_mv_img_grey_global_zs~=0),10)

            % calculate elastic registration from grey/TA reference global image with moving grey/TA reference global image
            %Dmv=calculate_elastic_registration(rf_img_grey_global,mv_img_grey_global,TA_rf_img_global,TA_mv_img_global,regE.szE,regE.bfE,regE.diE);
            Dmv=calculate_elastic_registration(norm_rf_img_grey_global,mv_img_grey_global,TA_rf_img_global,TA_mv_img_global,regE.szE,regE.bfE,regE.diE);

            % load previous transformation matrix
            %if exist([out_D,[reference_name(1:end-3),'mat']], 'file')
            %    load([out_D,[reference_name(1:end-3),'mat']], 'D');
            %else
                %temp = 'monkey_fetus_40_1104.tif';
                %load([matpth,temp(1:end-3),'mat'],'D');
                load([matpth,reference_name(1:end-3),'mat'],'D');
            %end
            D_og = D;
            %D=zeros(size(Dmv));

            % add elastic registration to transformation matrix
            D=D+Dmv;
            %D=Dmv;
            %Dmv2=imresize(Dmv,size(mv_img_grey)); % scale transformation matrix up to image size
            %immvE2=imwarp(mv_img_grey_global,Dmv2,'nearest','FillValues',fillval); % perform transformation on moving image

            % testing Dmv
            D=imresize(D,size(mv_img_grey)); % scale transformation matrix up to image size

            fillval=squeeze(mode(mode(mv_img,2),1))'; % get fill value for pixels that got warped
            %immvE=imwarp(mv_img_grey_global,D,'nearest','FillValues',fillval); % perform transformation on moving image

            %figure(68);imshowpair(immvE,rf_img_grey_global) % this isn't supposed to look aligned.
            % to check if it worked, go to ImageJ and get a stack of images
            % and put the new image (saved below) in it with the previous
            % image in the stack and then toggle back and forth
      
            % all that stuff was done on grey scale image, so this applies
            % the transformation matrix to the rgb image

            imout = register_global_im(mv_img,tform,cent,f,fillval); % global
            imout = imwarp(imout,D,'nearest','FillValues',fillval); % elastic

            % load ref im that was already registered
            if exist([fix_pth,reference_name(1:end-3),'jpg'], 'file')
                disp('loading')
                e_ref_im_grey = rgb2gray(imread([fix_pth,reference_name(1:end-3),'jpg']));
            else
                e_ref_im_grey = rgb2gray(imread([ereg_pth,reference_name(1:end-3),'jpg']));
            end
            figure(68);imshowpair(imout,e_ref_im_grey)
            %figure(69);imshowpair(TA_rf_img_global,TA_mv_img_global)

            % save
            disp('')
            % subplot(1,3,1);imshow(imwarp(mv_img_grey_global,D,'nearest','FillValues',fillval));subplot(1,3,2);imshow(imwarp(mv_img_grey_global,Dmv2,'nearest','FillValues',fillval));subplot(1,3,3);imshow(e_ref_im_grey)
            subplot(1,3,1);imshow(mv_img_grey_global);
            subplot(1,3,2);imshow(imwarp(mv_img_grey_global,D,'nearest','FillValues',fillval));
            subplot(1,3,3);imshow(e_ref_im_grey)
            imwrite(imout,['\\10.162.80.16\Andre_expansion\data\monkey_fetus\slicer_label_folders\fix_registration\' [moving_name(1:end-3),'jpg']])
            save([out_global,moving_name(1:end-3),'mat'],'tform','cent','szz','padall','regE','f');
            D = imresize(D, size(D_og,1:2));
            save([out_D,moving_name(1:end-3),'mat'],'D');
        end
    end
end

%%


